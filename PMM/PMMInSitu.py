"""
The module composed in this file is meant to facilitate both in-situ inverse
design and experimental control of plasma metamaterials composed of many
elements. The library is constructed for use with Longwei DC power supplies
which are connected on RS485 multidrop networks. When carrying out the
optimization fully in-situ, you must use a Rohde and Schwarz ZNB vector network
analyzer, or else you need to modify the vna commands and use a different VISA
library than RSInstrument. The VISA backend I use on my Macbook Pro is the 
National Instruments VISA. All that to say: this is NOT a general purpose
library and functions with a very specific experimental setup. For more
information, contact Jesse Rodriguez: jrodrig@stanford.edu
05/11/2023
"""

import minimalmodbus
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from RsInstrument import *
import serial
import glob
import sys
import time
import yaml
import threading
import math
from skopt import gp_minimize #bayes
from skopt.space import Real

###############################################################################
## Utility functions and globals
###############################################################################
c = 299792058
e = 1.60217662*10**(-19)
epso = 8.8541878128*10**(-12)
muo = 4*np.pi*10**(-7)
me = 9.1093837015*10**(-31)

def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def Demult_Obj_Comp(freq, S21, S31, f1, f2, df = 0.25, norms = []):
    """
    Compute the demultiplexer objective that is analogous to the computational
    inverse design library. f1 corresponds to port 2, f2 corresponds to port 3.

    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f1: first operating frequency in GHz
        f2: second operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i1_l = np.searchsorted(freq, f1-df/2, side='left')
    i1_r = np.searchsorted(freq, f1+df/2, side='right')
    i2_l = np.searchsorted(freq, f2-df/2, side='left')
    i2_r = np.searchsorted(freq, f2+df/2, side='right')
    T21 = np.power(10*np.ones_like(S21), S21/10)
    T31 = np.power(10*np.ones_like(S31), S31/10)

    correct_1 = np.sum(T21[i1_l:i1_r])
    incorrect_1 = np.sum(T31[i1_l:i1_r])
    correct_2 = np.sum(T31[i2_l:i2_r])
    incorrect_2 = np.sum(T21[i2_l:i2_r])

    if len(norms) > 0:
        c_1 = correct_1/norms[0]
        i_1 = incorrect_1/norms[1]
        c_2 = correct_2/norms[2]
        i_2 = incorrect_2/norms[3]
        return c_1*c_2 - i_1 - i_2, norms
    else:
        new_norms = [np.abs(correct_1), np.abs(incorrect_1),\
                     np.abs(correct_2), np.abs(incorrect_2)]
        return -1, new_norms


def Demult_Obj_dB(freq, S21, S31, f1, f2, df = 0.25, norms = []):
    """
    Compute the demultiplexer objective that focuses more instead on dB
    transmission values. f1 corresponds to port 2, f2 corresponds to port 3.

    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f1: first operating frequency in GHz
        f2: second operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i1_l = np.searchsorted(freq, f1-df/2, side='left')
    i1_r = np.searchsorted(freq, f1+df/2, side='right')
    i2_l = np.searchsorted(freq, f2-df/2, side='left')
    i2_r = np.searchsorted(freq, f2+df/2, side='right')
    T21 = np.power(10*np.ones_like(S21), S21/10)
    T31 = np.power(10*np.ones_like(S31), S31/10)
    DdB = S21-S31

    correct_1 = np.sum(T21[i1_l:i1_r])
    correct_2 = np.sum(T31[i2_l:i2_r])
    isolation_1 = np.sum(DdB[i1_l:i1_r])
    isolation_2 = np.sum(-DdB[i2_l:i2_r])

    if len(norms) > 0:
        c_1 = correct_1/norms[0]
        c_2 = correct_2/norms[1]
        i_1 = isolation_1/norms[2]
        i_2 = isolation_2/norms[3]
        return c_1*c_2 + 10*i_1 + 10*i_2, norms
    else:
        new_norms = [np.abs(correct_1), np.abs(correct_2),\
                     np.abs(isolation_1), np.abs(isolation_2)]
        c_1 = correct_1/new_norms[0]
        c_2 = correct_2/new_norms[1]
        i_1 = isolation_1/new_norms[2]
        i_2 = isolation_2/new_norms[3]
        return c_1*c_2 + 10*i_1 + 10*i_2, new_norms


def Waveguide_Obj_Comp(freq, S21, S31, f, df = 0.25, norms = []):
    """
    Compute the waveguide objective that is analogous to the computational
    inverse design library. Port 2 has to be correct port, otherwise switch
    S21 and S31.

    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f: operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i_l = np.searchsorted(freq, f-df/2, side='left')
    i_r = np.searchsorted(freq, f+df/2, side='right')
    T21 = np.power(10*np.ones_like(S21), S21/10)
    T31 = np.power(10*np.ones_like(S31), S31/10)

    correct = np.sum(T21[i_l:i_r])
    incorrect = np.sum(T31[i_l:i_r])

    if len(norms) > 0:
        c = correct/norms[0]
        i = incorrect/norms[1]
        return c - i, norms
    else:
        new_norms = [np.abs(correct), np.abs(incorrect)]
        return 0, new_norms

def Waveguide_Obj_Narrow(freq, S21, S31, f, df=0.25, norms=[],
                         w_in=1.0, w_oob=0.5):
    """
    Narrow-band objective: reward in-band to Port-2 and 
    penalize in-band Port-3 and out-of-band Port-2/Port-3 leakage.

    freq in GHz; Sxx in dB.
    
    Increase w_oob if wanting stricter.
    """
    import numpy as np
    i_l = np.searchsorted(freq, f - df/2, side='left')
    i_r = np.searchsorted(freq, f + df/2, side='right')

    T21 = np.power(10.0, S21/10.0)
    T31 = np.power(10.0, S31/10.0)

    # in-band
    in21 = np.sum(T21[i_l:i_r])
    in31 = np.sum(T31[i_l:i_r])

    # out-of-band (everything except the in-band slice)
    oob21 = np.sum(T21[:i_l]) + np.sum(T21[i_r:])
    oob31 = np.sum(T31[:i_l]) + np.sum(T31[i_r:])

    if len(norms) > 0:
        in21 /= norms[0]; in31 /= norms[1]; oob21 /= norms[2]; oob31 /= norms[3]
        return (w_in*in21) - (in31 + w_oob*(oob21 + oob31)), norms
    else:
        new_norms = [abs(in21), abs(in31), abs(oob21), abs(oob31)]
        # return a neutral value and the norms on first pass
        return 0.0, new_norms


def Waveguide_Obj_dB(freq, S21, S31, f, df = 0.25, norms = []):
    """
    Compute the waveguide objective that focuses more instead on dB
    isolation values. Port 2 has to be correct port, otherwise switch
    S21 and S31.
    
    Args:
        freq: frequency np.array, in GHz
        S21: S21 np.array, in dB
        S31: S31 np.array, in dB
        f: operating frequency in GHz
        df: bandwidth around operating frequencies in GHz
        norms: normalization factors
    """
    i_l = np.searchsorted(freq, f-df/2, side='left')
    i_r = np.searchsorted(freq, f+df/2, side='right')
    
    correct = np.sum(S21[i_l:i_r])
    incorrect = np.sum(S31[i_l:i_r])
    
    if len(norms) > 0:
        c = correct/norms[0]
        i = incorrect/norms[1]
        return c - i, norms
    else:
        new_norms = [np.abs(correct), np.abs(incorrect)]
        return 0, new_norms

# def save_progress_callback(rho_path, obj_path, pmm_instance, fpm, k, S, f, fwin, show, progress_dir):
#     """
#     After each NEW evaluation from gp_minimize:
#       - append rho and objective to CSVs
#       - save objective PDF (per call)
#       - run+save transmission PDF+CSV (per call)
#     """
#     def _callback(res):
#         import numpy as np, os

#         # 1) Append latest data (skopt minimizes => store positive objective)
#         new_rho = np.atleast_2d(res.x_iters[-1])
#         new_obj = np.array([-res.func_vals[-1]])
#         with open(rho_path, 'ab') as f_rho, open(obj_path, 'ab') as f_obj:
#             np.savetxt(f_rho, new_rho, delimiter=',')
#             np.savetxt(f_obj, new_obj, delimiter=',')

#         # 2) Read accumulated data (for plotting)
#         all_rhos = np.loadtxt(rho_path, delimiter=',')
#         all_objs = np.loadtxt(obj_path, delimiter=',')

#         if np.ndim(all_objs) == 0:
#             all_objs = np.array([float(all_objs)])
#         else:
#             all_objs = np.array(all_objs).flatten()

#         if np.ndim(all_rhos) == 1:
#             all_rhos = np.array([all_rhos])

#         call_idx = len(all_objs)
#         print(f"\n[Callback] Saving per-call plots (call {call_idx})")

#         # 3) Objective plot per call
#         obj_plot_path = obj_path.replace('.csv', f'_call{call_idx:03d}.pdf')
#         try:
#             pmm_instance.Plot_Obj(obj_plot_path, all_objs, show=False)
#         except Exception as e:
#             print(f"[WARN] Obj plot failed on call {call_idx}: {e}")

#         # 4) Transmission plot+csv per call (measure best-so-far)
#         try:
#             best_rho = all_rhos[np.argmax(all_objs)]
#             # This call writes a fixed filename first:
#             pmm_instance.Wvg_Run_And_Plot(
#                 progress_dir, best_rho, fpm, k, S, f, fwin=fwin, show=show
#             )
#             # Rename fixed filenames → per-call filenames
#             base = os.path.join(
#                 progress_dir, f"Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz_k{k:.1f}_S{S:.1f}"
#             )
#             src_pdf = base + ".pdf"
#             src_csv = base + ".csv"
#             dst_pdf = base + f"_call{call_idx:03d}.pdf"
#             dst_csv = base + f"_call{call_idx:03d}.csv"
#             if os.path.exists(src_pdf):
#                 os.replace(src_pdf, dst_pdf)
#                 print(f"    [✓] Spectrum PDF → {dst_pdf}")
#             if os.path.exists(src_csv):
#                 os.replace(src_csv, dst_csv)
#                 print(f"    [✓] Spectrum CSV → {dst_csv}")
#         except Exception as e:
#             print(f"[WARN] Spectrum plot/rename failed on call {call_idx}: {e}")

#     return _callback

def save_progress_callback(rho_path, obj_path, pmm_instance, fpm, k, S, f, fwin, show, progress_dir):
    """
    After each NEW evaluation from gp_minimize:
      - append rho and objective to CSVs
      - save objective PDF (per call)
      - run+save transmission PDF+CSV (per call)
      - PRINT per-bulb V/I and per-bulb Δ from last run
    """
    import os, numpy as np

    last_bulbset_path = os.path.join(progress_dir, "last_bulbset.npy")

    def _callback(res):
        # 1) Append latest data (skopt minimizes => store positive objective)
        new_rho = np.atleast_2d(res.x_iters[-1])
        new_obj = np.array([-res.func_vals[-1]])
        with open(rho_path, 'ab') as f_rho, open(obj_path, 'ab') as f_obj:
            np.savetxt(f_rho, new_rho, delimiter=',')
            np.savetxt(f_obj, new_obj, delimiter=',')

        # 2) Load all to determine call index
        all_rhos = np.loadtxt(rho_path, delimiter=',')
        all_objs = np.loadtxt(obj_path, delimiter=',')
        if np.ndim(all_objs) == 0: all_objs = np.array([float(all_objs)])
        if np.ndim(all_rhos) == 1: all_rhos = np.array([all_rhos])
        call_idx = len(all_objs)

        print(f"\n[Callback] Saving per-call plots (call {call_idx})")

        # === NEW: print per-bulb V/I and Δ since last call ===
        try:
            # Current bulb settings from rho -> fp -> (V,I)
            bulbset_now = pmm_instance.Rho_to_Bulb(
                new_rho[0], pmm_instance.f_a(fpm), knob=k, scale=S
            )  # shape (Nbulbs, 2) columns = [V, I]
            V_now = bulbset_now[:, 0]
            I_now = bulbset_now[:, 1]

            # Load previous bulbset if exists to compute deltas
            if os.path.exists(last_bulbset_path):
                bulbset_prev = np.load(last_bulbset_path)
                dV = V_now - bulbset_prev[:, 0]
                dI = I_now - bulbset_prev[:, 1]
            else:
                bulbset_prev = None
                dV = np.zeros_like(V_now)
                dI = np.zeros_like(I_now)

            # Print nicely (truncate if huge)
            np.set_printoptions(precision=3, suppress=True)
            print(f"[BulbSet] V (volts): {V_now.tolist()}")
            print(f"[BulbSet] I (amps):  {I_now.tolist()}")
            if bulbset_prev is None:
                print("[BulbSet] ΔV/ΔI: (first call → no previous state; deltas shown as 0.0)")
            else:
                print(f"[BulbSet] ΔV: {dV.tolist()}")
                print(f"[BulbSet] ΔI: {dI.tolist()}")

            # Save current bulbset for next-call delta
            np.save(last_bulbset_path, bulbset_now)
        except Exception as e:
            print(f"[WARN] Could not compute/print bulb settings: {e}")

        # 3) Objective plot per call
        obj_plot_path = obj_path.replace('.csv', f'_call{call_idx:03d}.pdf')
        try:
            pmm_instance.Plot_Obj(obj_plot_path, all_objs, show=False)
        except Exception as e:
            print(f"[WARN] Obj plot failed on call {call_idx}: {e}")

        # 4) Transmission plot+csv per call (measure best-so-far)
        try:
            best_rho = all_rhos[np.argmax(all_objs)]
            pmm_instance.Wvg_Run_And_Plot(
                progress_dir, best_rho, fpm, k, S, f, fwin=fwin, show=show  # show=True will pop the plot
            )
            base = os.path.join(
                progress_dir, f"Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz_k{k:.1f}_S{S:.1f}"
            )
            src_pdf = base + ".pdf";  src_csv = base + ".csv"
            dst_pdf = base + f"_call{call_idx:03d}.pdf"
            dst_csv = base + f"_call{call_idx:03d}.csv"
            if os.path.exists(src_pdf):
                os.replace(src_pdf, dst_pdf);  print(f"    [✓] Spectrum PDF → {dst_pdf}")
            if os.path.exists(src_csv):
                os.replace(src_csv, dst_csv);  print(f"    [✓] Spectrum CSV → {dst_csv}")
        except Exception as e:
            print(f"[WARN] Spectrum plot/rename failed on call {call_idx}: {e}")

    return _callback




###############################################################################
## In-situ inverse design class
###############################################################################
class PMMInSitu:
    # def __init__(self, conf_file, conf_dir = './../confs/'):
    #     with open(conf_file, 'r') as conf:
    #         self.config = yaml.load(conf, Loader=yaml.SafeLoader)

    #     self.a = self.config['array-a']
    #     self.mu = self.config['mobility']
    #     self.L = self.config['bulb-length']
    #     self.VtoI = np.loadtxt(conf_dir+'VtoI.txt', delimiter = ',')
    #     self.bulbs = {'all': {}}
    #     self.VNA = self.config['VNA']
    #     ports = serial_ports()
    #     for port in self.config['serial_ports']:
    #         if port not in ports:
    #             raise RuntimeError("One or more of the ports in the config file\
    #                                 is not connected")
                
    #         print("checking port", port)
    #         # Create 'all' pathway
    #         self.bulbs['all'][port] = minimalmodbus.Instrument(port = port,\
    #                                   slaveaddress = 0,\
    #                                   mode = minimalmodbus.MODE_RTU)
    #         self.bulbs['all'][port].serial.baudrate = 9600
    #         self.bulbs['all'][port].serial.bytesize = 8
    #         self.bulbs['all'][port].serial.parity = minimalmodbus.serial.PARITY_NONE
    #         self.bulbs['all'][port].serial.stopbits = 1
    #         self.bulbs['all'][port].serial.timeout = 1
    #         self.bulbs['all'][port].serial.close_port_after_each_call = True
    #         self.bulbs['all'][port].serial.clear_buffers_before_each_transaction = True

    #         # Create bulb entries in dict
    #         for bulb_addr in self.config['serial_ports'][port]:
    #             self.bulbs[bulb_addr] = {'I': 0.0, 'V': 0.0,\
    #                                 'Inst': minimalmodbus.Instrument(\
    #                                 port = port, slaveaddress = bulb_addr,\
    #                                 mode = minimalmodbus.MODE_RTU)}
    #             self.bulbs[bulb_addr]['Inst'].serial.baudrate = 9600
    #             self.bulbs[bulb_addr]['Inst'].serial.bytesize = 8
    #             self.bulbs[bulb_addr]['Inst'].serial.parity = minimalmodbus.serial.PARITY_NONE
    #             self.bulbs[bulb_addr]['Inst'].serial.stopbits = 1
    #             self.bulbs[bulb_addr]['Inst'].serial.timeout = 1
    #             self.bulbs[bulb_addr]['Inst'].serial.close_port_after_each_call = True
    #             self.bulbs[bulb_addr]['Inst'].serial.clear_buffers_before_each_transaction = True

    #             # Now ping each of the power supplies and make sure they are
    #             # connected to the correct RS-485 bus and aren't already running.
    #             try:
    #                 on = self.bulbs[bulb_addr]['Inst'].read_register(\
    #                         registeraddress=0x1004)
    #                 if on == 1:
    #                     print("The power supply associated with bulb "\
    #                           +str(bulb_addr)+" is putting out power, fixing "+\
    #                           "now. Check power supply.")
    #                     time.sleep(1.0)
    #                     self.bulbs[bulb_addr]['Inst'].write_register(\
    #                             registeraddress=0x1006, value = 0, functioncode = 6)
    #             except:
    #                 raise RuntimeError("Bulb "+str(bulb_addr)+" is not connected "+\
    #                                    "to the correct RS-485 bus.")
    
    def _parallel_check_bulb(self, addr, verbose=False):
        """
        Helper used by __init__ to ping each bulb in parallel:
        - reads the ON/OFF register
        - if a supply is on, turn it off
        - raises a clear error if the bulb isn't reachable
        """
        import time
        try:
            inst = self.bulbs[addr]['Inst']
            on = inst.read_register(registeraddress=0x1004)  # 1 = ON, 0 = OFF
            if on == 1:
                if verbose:
                    print(f"Bulb {addr} was ON at init; turning OFF.")
                time.sleep(1.0)
                inst.write_register(registeraddress=0x1006, value=0, functioncode=6)  # OFF
        except Exception as e:
            raise RuntimeError(
                f"Bulb {addr} is not connected to the correct RS-485 bus (init check failed): {e}"
            )

    
    
    def __init__(self, conf_file, conf_dir = './../confs/', verbose=False):
        self.verbose = verbose  

        with open(conf_file, 'r') as conf:
            self.config = yaml.load(conf, Loader=yaml.SafeLoader)

        self.a = self.config['array-a']
        self.mu = self.config['mobility']
        self.L = self.config['bulb-length']
        self.VtoI = np.loadtxt(conf_dir+'VtoI.txt', delimiter = ',')
        self.bulbs = {'all': {}}
        self.VNA = self.config['VNA']
        ports = serial_ports()
        for port in self.config['serial_ports']:
            if port not in ports:
                raise RuntimeError("One or more of the ports in the config file\
                                    is not connected")
            if self.verbose:
                print("checking port", port)

            # Create 'all' pathway
            self.bulbs['all'][port] = minimalmodbus.Instrument(port = port,\
                                    slaveaddress = 0,\
                                    mode = minimalmodbus.MODE_RTU)
            self.bulbs['all'][port].serial.baudrate = 9600
            self.bulbs['all'][port].serial.bytesize = 8
            self.bulbs['all'][port].serial.parity = minimalmodbus.serial.PARITY_NONE
            self.bulbs['all'][port].serial.stopbits = 1
            self.bulbs['all'][port].serial.timeout = 1
            self.bulbs['all'][port].serial.close_port_after_each_call = True
            self.bulbs['all'][port].serial.clear_buffers_before_each_transaction = True

            # Create bulb entries in dict
            for bulb_addr in self.config['serial_ports'][port]:
                self.bulbs[bulb_addr] = {'I': 0.0, 'V': 0.0,\
                                    'Inst': minimalmodbus.Instrument(\
                                    port = port, slaveaddress = bulb_addr,\
                                    mode = minimalmodbus.MODE_RTU)}
                self.bulbs[bulb_addr]['Inst'].serial.baudrate = 9600
                self.bulbs[bulb_addr]['Inst'].serial.bytesize = 8
                self.bulbs[bulb_addr]['Inst'].serial.parity = minimalmodbus.serial.PARITY_NONE
                self.bulbs[bulb_addr]['Inst'].serial.stopbits = 1
                self.bulbs[bulb_addr]['Inst'].serial.timeout = 1
                self.bulbs[bulb_addr]['Inst'].serial.close_port_after_each_call = True
                self.bulbs[bulb_addr]['Inst'].serial.clear_buffers_before_each_transaction = True

        # Now ping each of the power supplies in parallel.
        self.parallel_bulb_op('_parallel_check_bulb')



    def _cmd_with_retry(self, cmd, addr, cmd_args, cmd_kwargs, tries=3, delay=0.6, verbose=None):
        """Call a bulb command and retry on any exception."""
        import time
        v = self.verbose if (verbose is None) else verbose
        for attempt in range(tries):
            try:
                cmd(addr, *cmd_args, **cmd_kwargs)
                return True
            except Exception as err:
                if attempt == tries - 1:
                    if v:
                        print(f"[FAIL] Addr {addr}: {cmd.__name__} → {err}")
                else:
                    if v:
                        print(f"[WARN] Addr {addr}: {cmd.__name__} failed ({attempt+1}/{tries}); retrying…")
                    time.sleep(delay)
        return False



    # def parallel_bulb_op(self, cmd_name, *cmd_args, tries=3, delay=0.6, verbose=None, **cmd_kwargs):
    #     """
    #     Runs a command on all bulbs, processing each serial port's list of bulbs in parallel
    #     using the 'threading' module.
    #     """
    #     v = self.verbose if (verbose is None) else verbose
    #     cmd = getattr(self, cmd_name)
    #     port_to_addrs = self.config["serial_ports"]

    #     def process_one_port(address_list):
    #         for addr in address_list:
    #             self._cmd_with_retry(cmd, addr, cmd_args, cmd_kwargs, tries, delay, verbose=v)

    #     threads = []
    #     if v:
    #         print(f"Starting threads for {len(port_to_addrs)} ports...")
    #     for addr_list in port_to_addrs.values():
    #         thread = threading.Thread(target=process_one_port, args=(addr_list,))
    #         threads.append(thread)
    #         thread.start()
    #     if v:
    #         print("Waiting for all threads to complete...")
    #     for thread in threads:
    #         thread.join()
    #     if v:
    #         print(f"Finished '{cmd_name}' on all ports.")
    #     return
    def parallel_bulb_op(self, cmd_name, *cmd_args, tries=3, delay=0.6, **cmd_kwargs):
        """
        Run a command across RS-485 buses in parallel.

        """
        import threading, time
        cmd = getattr(self, cmd_name)
        port_to_addrs = self.config["serial_ports"]

        def process_one_bus(port, address_list):
            # Always broadcast for Activate/Deactivate
            if cmd_name in ("Activate_Bulb", "Deactivate_Bulb"):
                inst = self.bulbs['all'][port]  # broadcast instrument (slaveaddress=0)
                value = 1 if cmd_name == "Activate_Bulb" else 0
                for attempt in range(tries):
                    try:
                        inst.write_register(registeraddress=0x1006, value=value, functioncode=6)
                        return
                    except Exception:
                        if attempt < tries - 1:
                            time.sleep(delay)
                return
            # Otherwise, do per-bulb ops (serial per bus, parallel across buses)
            for addr in address_list:
                self._cmd_with_retry(cmd, addr, cmd_args, cmd_kwargs, tries, delay)

        threads = []
        for port, addr_list in port_to_addrs.items():
            t = threading.Thread(target=process_one_bus, args=(port, addr_list))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

           
    
    # def Address(self, coords):
    #     """
    #     Takes array coordinates and returns a bulb address. 
    #     THIS IS THE OLD VERSION.

    #     Args:
    #         coords: tuple/list; e.g. (i,j)
    #     """
    #     return (coords[0]+self.config['array-x']*coords[1])
    
    def Address(self, radius):
        """
        Generates axial coordinates for a hexagon of a given radius.
        The radius determines the number of bulbs on each side.
        For 6 bulbs per side, radius = 5.
        """
        coords = []
        for r in range(-radius, radius + 1):
            for q in range(max(-radius, -r - radius), min(radius, -r + radius) + 1):
                coords.append((q, r))
        return coords

    def plot_Address(self, radius=5, pause=0.03):
        """
        Probes the status of each bulb and plots them on a hexagonal grid.
        A radius of 5 corresponds to a hexagon with 6 bulbs per side (91 total).
        """
        #PMM.plot_Addresss(radius=5)
        
        # Get all bulb addresses in the order they appear in the config file
        all_addrs = [addr for port_addrs in self.config['serial_ports'].values() for addr in port_addrs]
        
        coords = self.Address(radius)
        num_bulbs_in_hex = len(coords)
        
        if len(all_addrs) < num_bulbs_in_hex:
            print(f"Warning: Config file has {len(all_addrs)} bulbs, but a hexagon of radius {radius} requires {num_bulbs_in_hex}.")
            num_bulbs_in_hex = len(all_addrs)

        addrs_to_plot = all_addrs[:num_bulbs_in_hex]
        addr_coord_map = dict(zip(addrs_to_plot, coords))

        # --- Probe bulb status ---
        on_set = set()
        print(f"Checking status of {num_bulbs_in_hex} bulbs...")
        for addr in addrs_to_plot:
            try:
                inst = self.bulbs[addr]['Inst']
                if inst.read_register(0x1004) & 0x0001: # Check ON/OFF 
                    on_set.add(addr)
            except Exception as e:
                # This can happen if a bulb is unresponsive
                pass 
            time.sleep(pause)

        xs, ys, labels, colors = [], [], [], []
        for addr, (q, r) in addr_coord_map.items():
            # Convert axial hex coords to cartesian (x,y) for plotting
            x = (q + r / 2.0) * np.sqrt(3)
            y = (3.0 / 2.0) * r
            xs.append(x)
            ys.append(y)
            labels.append(addr)
            colors.append('black' if addr in on_set else 'white')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(xs, ys, s=500, c=colors, edgecolors='black', linewidths=1.5)

        for i, label in enumerate(labels):
            text_color = 'white' if colors[i] == 'black' else 'black'
            ax.text(xs[i], ys[i], str(label), ha='center', va='center', color=text_color, fontsize=8, weight='bold')

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{num_bulbs_in_hex}-Bulb Hexagon (Side Length: {radius+1})", fontsize=16)
        plt.tight_layout()
        plt.show()


    def Set_Bulb_VI(self, addr, V, I, verbose = True):
        """
        Set bulb current and voltage

        Args:
            addr: int, bulb address
            V: float, bulb voltage in [0,20] (volts)
            I: float, bulb current in [0,10] (amps)
        """
        if addr == 'all':
            if verbose:
                print("WARNING: Setting the current and voltage of all bulbs at "+\
                  "once invalidates the tracking of I and V in the bulb dict. "+\
                  "Do not query V or I until the bulbs are set individually "+\
                  "again.")
            for port in self.bulbs['all']:
                self.bulbs['all'][port].write_register(registeraddress = 0x1000,\
                value = V*100, functioncode = 6)
                time.sleep(0.005)
                self.bulbs['all'][port].write_register(registeraddress = 0x1001,\
                value = I*100, functioncode = 6)
        else:
            self.bulbs[addr]['V'] = V
            self.bulbs[addr]['I'] = I
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1000,\
                    value = V*100, functioncode = 6)
            time.sleep(0.005)
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1001,\
                    value = I*100, functioncode = 6)
        return
    
    
    def Run_Bulb_VI(self, addr, V, I, t = 0, verbose = True):
        """
        Sets and activates the bulb for t seconds using the proper procedure.
        
        Args:
            addr: int, bulb address
            V: float, bulb voltage in [0,20] (volts)
            I: float, bulb current in [0,10] (amps)
            t: float, time to stay activated (seconds). Default is to stay on
               indefinitely.
        """
        self.Set_Bulb_VI(addr, 12, 10, verbose)
        self.Activate_Bulb(addr)
        time.sleep(0.3)
        self.Set_Bulb_VI(addr, V, I, verbose)
        
        if t > 0.005:
            time.sleep(t)
            self.Deactivate_Bulb(addr)
        
        return


    def Config_Check(self):
        """
        Sets the current and voltage of every power supply to be equal to their
        bulb address and activates them to make sure each supply is turned on
        and the RS-485 bus is connected properly.
        """
        for addr in self.bulbs:
            if addr != 'all':
                self.Set_Bulb_VI(addr, addr/100, 0)

        self.Activate_Bulb('all')

        return


    def Config_Warmup(self, T = 10, ballasts = 'New', duty_cycle = 0.5):
        """
        Runs the standard warm-up procedure for the bulb array
        """
        if ballasts == 'New':
            activate = 20
        else:
            activate = 12
        for i in range(T):
            print("Warmup cycle", i+1)
            self.Set_Bulb_VI('all', activate, 10, verbose = False)
            time.sleep(0.5)
            self.Activate_Bulb('all')
            time.sleep(4)
            self.Set_Bulb_VI('all', activate-4, 10, verbose = False)
            time.sleep(10)
            self.Deactivate_Bulb('all')
            time.sleep(15/duty_cycle-15)

        return


    # def Activate_Bulb(self, addr):
    #     """
    #     Activate bulb
    #     """
    #     if addr == 'all':
    #         for port in self.bulbs['all']:
    #             self.bulbs['all'][port].write_register(registeraddress = 0x1006,\
    #                     value = 1, functioncode = 6)
    #     else:
    #         self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,\
    #                 value = 1, functioncode = 6)
    #     return

    def Activate_Bulb(self, addr):
        """
        Activate bulb
        """
        if addr == 'all':
            return self.parallel_bulb_op("Activate_Bulb", tries=3, delay=0.6)
        else:
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,
                                                    value = 1, functioncode = 6)
        return



    # def Deactivate_Bulb(self,addr):
    #     """
    #     Deactivate bulb
    #     """
    #     if addr == 'all':
    #         for port in self.bulbs['all']:
    #             self.bulbs['all'][port].write_register(registeraddress = 0x1006,\
    #                     value = 0, functioncode = 6)
    #     else:
    #         self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,\
    #                 value = 0, functioncode = 6)
    #     return

    def Deactivate_Bulb(self, addr, parallel=True):
        """
        Deactivate bulb
        """
        if addr == 'all':
            if parallel:
                return self.parallel_bulb_op("Deactivate_Bulb", tries=3, delay=0.6)
            else:
                self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,
                                                    value = 0, functioncode = 6)
                return
        else:
            self.bulbs[addr]['Inst'].write_register(registeraddress = 0x1006,
                                                    value = 0, functioncode = 6)
        return

    # def Check_Power_Healthy(self, min_v=6.5, min_i=0.5, pause=0.0, verbose=None):
    #     """
    #     Verifies each supply is ON and has non-trivial set V/I.
    #     Returns (ok, bad_list). bad_list = [(addr, on, V, I), ...].
    #     """
    #     v = self.verbose if (verbose is None) else verbose
    #     bad = []
    #     for addr in self.bulbs:
    #         if addr == 'all':
    #             continue
    #         try:
    #             inst = self.bulbs[addr]['Inst']
    #             on = inst.read_register(0x1004)
    #             setV = inst.read_register(0x1000) / 100.0
    #             setI = inst.read_register(0x1001) / 100.0
    #             if (on != 1) or (setV < min_v) or (setI < min_i):
    #                 bad.append((addr, on, setV, setI))
    #         except Exception:
    #             bad.append((addr, -1, -1.0, -1.0))  # unreadable = failure
    #         if pause > 0:
    #             time.sleep(pause)
    #     if v and bad:
    #         print("[Check_Power_Healthy] FAIL:", bad)
    #     return (len(bad) == 0), bad



    def Scale_Rho_ne(self, rho, wp_max):
        """
        Uses an arctan barrier to map optimal parameters from the computational 
        inverse design library to plasma density values (dimensionalized, m^-3)

        Args:
            rho: Parameters being optimized
            wp_max: Approximate maximum non-dimensionalized plasma frequency
        """
        wp = (wp_max/1.5)*np.arctan(rho/(wp_max/7.5))
        wp_dim = wp*c/self.a*2*np.pi
        ne = wp_dim**2*me*epso/e**2

        return ne

    def Scale_Rho_fp_legacy(self, rho, wp_max):
        """
        Uses an arctan barrier to map optimal parameters from the computational 
        inverse design library to plasma frequency values (dimensionalized, GHz)
        
        USE THIS WHEN TESTING SIMULATION PARAMETERS.

        Args:
            rho: Parameters being optimized
            wp_max: Approximate maximum non-dimensionalized plasma frequency
        """
        fp = (wp_max/1.5)*np.arctan(np.abs(rho)/(wp_max/7.5))
        fp_dim = fp*c/self.a/10**9

        return fp_dim


    def Scale_Rho_fp(self, rho, wp_max):
        """
        Maps optimal parameters from the computational 
        inverse design library to plasma frequency values (dimensionalized, GHz).
        
        Args:
            rho: Parameters being optimized
            wp_max: Approximate maximum non-dimensionalized plasma frequency
        """
        fp_nd = np.clip(rho, 0.0, None) # making it only positive
        #fp_nd = (wp_max / 1.2) * np.arctan(rho_pos / (wp_max / 10)) #use for
        fp_dim_GHz = fp_nd * c / self.a / 1e9 # convert to GHz
        ceiling = getattr(self, "fp_ceiling_GHz", 20.0)  # hardware safety limit
        
        # ----- verbose prints -----
        if self.verbose:
            if hasattr(fp_dim_GHz, "size"):
                n = int(fp_dim_GHz.size)
                rho_min = float(np.min(rho))
                rho_med = float(np.median(rho))
                rho_max = float(np.max(rho))
                fp_min  = float(np.min(fp_dim_GHz))
                fp_med  = float(np.median(fp_dim_GHz))
                fp_max  = float(np.max(fp_dim_GHz))
                clipped_hi = int(np.sum(fp_dim_GHz > ceiling))
                clipped_lo = int(np.sum(rho < 0))
            else:
                n = 1
                rho_min = rho_med = rho_max = float(rho)
                fp_min  = fp_med  = fp_max  = float(fp_dim_GHz)
                clipped_hi = int(fp_dim_GHz > ceiling)
                clipped_lo = int(rho < 0)
            print(
                "[Scale_Rho_fp]",
                f"size={n}",
                f"rho[min,med,max]=({rho_min:.4g},{rho_med:.4g},{rho_max:.4g})",
                f"fp_GHz[min,med,max]=({fp_min:.4g},{fp_med:.4g},{fp_max:.4g})",
                f"clipped_hi={clipped_hi} clipped_lo={clipped_lo} ceil={ceiling} GHz"
            )
        
        
        return np.clip(fp_dim_GHz, 0.0, ceiling)

        
    @staticmethod
    def BulbSetting_BOLSIG(fp, knob = 0.5, scale = 1.0): #add self back in when not static
            """
            Maps plasma frequency value in GHz to a current and voltage setting for
            the DC power supplies using updated linear (current) and log (voltage) fits:
            Current: fp(I) = S * ( (0.6 + 0.65*k) + 0.2*I )
            Voltage: fp(V) = S * ( (10.5 + 2.5*k) * log5(V - 4.8) - 4.5 )

            Args:
                fp: plasma frequency in GHz 
                knob: constant to tune experimental fit. knob = 0 is low end
                    (bottom curve) and knob = 1 is high end (top curve).
                scale: Parameter that scales the overall plasma frequency values.
            """
            k = knob
            S = scale

            I_min, I_max = 0.1, 10.0 # Amps
            V_min, V_max = 6.0, 20.0 # Volts

            Min_curr = S * ( (0.6 + 0.65*k) + 0.2*I_min ) #simplified
            Max_curr = S * ( (0.6 + 0.65*k) + 0.2*I_max )

            Min_volt = S * ( (10.5 + 2.5*k) * np.log(V_min - 4.8) / np.log(5) - 4.5 )
            Max_volt = S * ( (10.5 + 2.5*k) * np.log(V_max - 4.8) / np.log(5) - 4.5 )

            if fp < Min_curr/2:
                return (0,0)

            elif fp >= Min_curr/2 and fp < Min_curr:
                # Ignition: Set to max voltage and calculate current for Min_curr boundary
                I = (Min_curr/S - (0.6 + 0.65*k)) / 0.2 # Inverse of Current
                return (20, I) # Use new max voltage

            elif fp >= Min_curr and fp < Max_curr: 
                # Current-controlled region: V is fixed, I is varied
                I = (fp/S - (0.6 + 0.65*k)) / 0.2
                return (20, I) # Use new max voltage

            elif fp >= Max_curr and fp < Min_volt: # this is IMPOSSIBLE
                # Transition region between max current and min voltage
                if fp < Max_curr+(Min_volt-Max_curr)/2:
                    # Lower half: stay at max current setting
                    I = (Max_curr/S - (0.6 + 0.65*k)) / 0.2
                    return (20, I) # Use new max voltage
                else:
                    # Upper half: switch to max current and calculate voltage for Min_volt boundary
                    V = np.power(5, (Min_volt/S + 4.5)/(10.5 + 2.5*k)) + 4.8 # Inverse of voltage
                    return (V, 10) # Use max current

            elif fp >= Min_volt and fp <= Max_volt:
                # Voltage-controlled region: I is fixed, V is varied
                V = np.power(5, (fp/S + 4.5)/(10.5 + 2.5*k)) + 4.8
                return (V, 10) # Use max current

            elif fp > Max_volt:
                # Saturation: return max settings
                return (20, 10) # Use new max voltage

    def BulbSetting_BOLSIG_Fix(self, fp, knob=0.5, scale=1.0):
        """
        Voltage-only version of BulbSetting_BOLSIG.
        """
        k = knob
        S = scale

        V_min = 7.0
        V_max = 20.0
        I_fixed = 10.0

        # fp(V) = S * ( (10.5 + 2.5*k) * log5(V - 4.8) - 4.5 )
        Min_volt = S * ((10.5 + 2.5*k) * np.log(V_min - 4.8) / np.log(5) - 4.5)
        Max_volt = S * ((10.5 + 2.5*k) * np.log(V_max - 4.8) / np.log(5) - 4.5)

        if fp <= Min_volt:
            return (0.0, 0.0)  # <-- only change...maybe make this small not zero? maybe i should go back to including current stuff...
            # return(7.0, 1.0)
        elif fp <= Max_volt:
            # inverse of the same voltage fit:
            # V(fp) = 5^((fp/S + 4.5)/(10.5 + 2.5*k)) + 4.8
            V = np.power(5.0, (fp / S + 4.5) / (10.5 + 2.5*k)) + 4.8
            V = float(np.clip(V, V_min, V_max))
            return (V, I_fixed)
        else:
            return (V_max, I_fixed)


    
    def Rho_to_Bulb(self, rho, wp_max, knob = 0.5, scale = 1.0,\
                    ballast = 'New'):
        """
        Accepts optimal parameter array (MUST BE FLATTENED) and returns (V,I)
        for each bulb.

        Args:
            rho: optimal parameter array (flattened) from PMMInverse library
            wp_max: Approximate maximum non-dimensionalized plasma frequency
            knob: constant to tune experimental fit to lower and upper range of
                  BOLSIG cases. knob = 0 is low end and knob = 1 is high end.
            scale: Parameter that scales the overall plasma frequency values.
        """
        BulbSet = np.zeros((rho.shape[0],2))
        fp = self.Scale_Rho_fp(rho, wp_max)

        for i in range(rho.shape[0]):
            if ballast == 'New':
                BulbSet[i,:] = self.BulbSetting_BOLSIG_Fix(fp[i], knob, scale)
            else:
                BulbSet[i,:] = self.BulbSetting_BOLSIG_Fix(fp[i], knob, scale)

        return BulbSet

    
    def ArraySet_Rho(self, rho, wp_max, knob = 0.5, scale = 1.0, ballast = 'New', verbose=None):
        """
        Accepts optimal parameter array (MUST BE FLATTENED) and activates the
        bulb array accordingly.
        """
        v = self.verbose if (verbose is None) else verbose

        BulbSet = self.Rho_to_Bulb(rho, wp_max, knob, scale, ballast)
        activate = 14 if ballast == 'New' else 12

        self.Set_Bulb_VI('all', activate, 10, verbose=False)
        time.sleep(0.005)
        self.Activate_Bulb('all')  
        time.sleep(1)
        self.Set_Bulb_VI('all', activate-4, 10, verbose=False)

        port_to_addrs = self.config["serial_ports"]
        
        def process_port_for_set_vi(address_list, bulb_settings):
            for addr in address_list:
                bulb_index = addr - 1
                V = bulb_settings[bulb_index, 0]
                I = bulb_settings[bulb_index, 1]
                
                not_set = True
                tries = 0
                while not_set and tries < 5:
                    try:
                        self.Set_Bulb_VI(addr, V, I, verbose=False)
                        time.sleep(0.005)
                        not_set = False
                    except:
                        tries += 1
                        if v:
                            print('Trouble setting bulb '+str(addr)+', trying again')
                        time.sleep(1)
                if not_set:
                    try:
                        self.Set_Bulb_VI(addr, V, I, verbose=False)
                        time.sleep(0.005)
                    except:
                        # safety shutdown unchanged
                        self.Deactivate_Bulb('all')
                        time.sleep(3)
                        self.Deactivate_Bulb('all')
                        raise RuntimeError("Failed to set bulb "+str(addr)+\
                                        " six times. Check config.")

        threads = []
        for addr_list in port_to_addrs.values():
            thread = threading.Thread(target=process_port_for_set_vi, args=(addr_list, BulbSet))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return

    


    def Get_S21_S31(self):
        """
        Gets the freq array, S21 and S31 from the R&S VNA. Make sure the VNA is
        in the measurement state you want PRIOR to running this function. In 
        our case, that is with our cal set, 10000 points, Avg. factor 10.
        """
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                instr = RsInstrument(self.VNA)

                instr.write_str('TRIGger1:SEQuence:SOURce IMM')
                time.sleep(7)
                S21_str = instr.query_str('CALC1:DATA:TRAC? "Trc1", FDAT')
                S31_str = instr.query_str('CALC1:DATA:TRAC? "Trc2", FDAT')
                freq_str = instr.query_str('CALC1:DATA:STIM?')
                
                instr.write_str('TRIGger1:SEQuence:SOURce MAN')
                instr.close()
                
                # Check for valid, non-empty responses before converting
                if S21_str and S31_str and freq_str:
                    S21 = np.array(S21_str.split(','), dtype=float)
                    S31 = np.array(S31_str.split(','), dtype=float)
                    freq = np.array(freq_str.split(','), dtype=float)
                    return freq, S21, S31
                else:
                    raise ValueError("VNA returned empty data.")

            except Exception as e:
                print(f"Warning: VNA communication failed on attempt {attempt + 1}/{max_attempts}. Error: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait for 2 seconds before retrying
                else:
                    # print("Error: VNA communication failed after multiple attempts. Returning empty arrays.")
                    # return np.array([]), np.array([]), np.array([])
                    raise RuntimeError("VNA communication failed after multiple attempts. Check VNA connection and VISA settings.")

    def Test_Comp_Rho(self, rho, fpm, k_S = [], f_op = [], fwin = [],\
                      wu = 10, save_dir = './', duty_cycle = 0.5, show = True):
        """
        Takes optimal rho from the computational inverse design library and 
        performs a test that sweeps through a user-defined sweep of the fit
        parameters k and S

        Args:
            rho: np.array, optimal parameters
            fpm: float, max plasma frequency in GHz
            k_S: np.array, [k,S] pairs for each iteration of the test procedure.
            f_op: list of operating frequencies in GHz
            fwin: list, plotting window for frequency
            save_dir: str, directory to save in
            duty_cycle: float in [0,1], duty cycle of array
            show: bool, show plot or not.
        """
        print("="*80)
        print("Running array warmup.")
        print("="*80)
        self.Config_Warmup(T = wu, ballasts = 'New', duty_cycle = duty_cycle)
        print("\n")
        print("="*80)
        print("Array warm! Beginning test.")
        print("="*80)
        
        for i in range(k_S.shape[0]):
            print('Running k = %.1f, S = %.1f case.'%(k_S[i,0], k_S[i,1]))
            if len(f_op) > 1:
                self.Demult_Run_And_Plot(save_dir, rho, fpm, k_S[i,0], k_S[i,1],\
                            f_op[0], f_op[1], fwin = fwin, show = show)
            else:
                self.Wvg_Run_And_Plot(save_dir, rho, fpm, k_S[i,0], k_S[i,1],\
                            f_op[0], fwin = fwin, show = show)
            time.sleep(20/duty_cycle-22)

        return




    def Optimize_Demultiplexer(self, epochs, rho, fpm, k, S, f1, f2, df = 0.25,\
                               alpha = 0.01, sample = 12, p = 0.1,\
                               objective = 'comp', optimizer = 'grad. asc.',\
                               wu = 10, progress_dir = '.', fwin = [],\
                               duty_cycle = 0.5, show = True,\
                               restart_obj = False, verbose = False,\
                               ID = ''):
        """
        Performs an in-situ optimization procedure to produce a demultiplexer that 
        differentiates between freqeuncies f1 and f2.

        Args:
            epochs: int, Number of epochs (1 epoch = all bulbs adjusted once)
            rho: np.array, Starting parameters
            fpm: float, Max plasma frequency in GHz
            k: float in [0,1], Bulb fit knob
            S: float in [0,1], Bulb fit scale factor
            f1: float, frequency 1 in GHz
            f2: float, frequency 2 in GHz
            df: float, bandwidth around operating frequencies in GHz
            alpha: float, learning rate
            sample: int, How many bulbs at a time are modified to compute 
                    gradient wrt subset of parameters
            p: float, std. dev. of noise added to parameters
            objective: str, chooses objective function
            wu: int, # of minutes to warm up the array
        """
        if os.path.isfile(progress_dir+'/rho_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'\
                %(f1,f2,fpm)+ID+'.csv'):
            obj = self.Read_Params(progress_dir+\
                    '/obj_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv').tolist()
            norms = self.Read_Params(progress_dir+\
                    '/norms_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv').tolist()
            rho_evolution = self.Read_Params(progress_dir+\
                    '/rho_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv')
            rho = np.copy(rho_evolution[np.argmax(obj),:])
            print('='*80)
            print('NOTE: Optimizer starting over from sample '+\
                    '%d of previous run'%(np.argmax(obj)+1))
            print('='*80)
            continuation = True
        else:
            rho_evolution = np.zeros((1,rho.shape[0]))
            rho_evolution[0,:] = np.copy(rho)
            continuation = False

        num_bulbs = rho.shape[0]
        bulb_idx = np.array(list(range(num_bulbs)))

        if num_bulbs%sample != 0:
            per_epoch = num_bulbs//sample
        else:
            per_epoch = num_bulbs//sample
        
        print("="*80)
        print("Initiating demultiplexer optimization. You have chosen to run "+\
              str(epochs)+" epochs with a\nsample factor of "+str(sample)+".")
        print("Since there are "+str(num_bulbs)+" bulbs, this means that each epoch"+\
              " will take %.1f minutes, for\na total runtime of"%(per_epoch*38/60)+\
              " about %.1f minutes."%(per_epoch*epochs+wu))
        print("="*80)
        print("\n")
        print("="*80)
        print("Running array warmup.")
        print("="*80)
        self.Config_Warmup(T = wu, ballasts = 'New', duty_cycle = duty_cycle)

        print("\n")
        print("="*80)
        print("Array warm! Beginning optimization.")
        print("="*80)

        if not continuation:
            obj = []
            t1 = time.time()
            o, norms = self.Demult_Obj_Get(rho, fpm, k, S, f1, f2, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
            obj.append(o)
            t2 = time.time()

            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
            print("="*80)
            self.Save_Params(np.array(norms), progress_dir+\
                    '/norms_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv')
        else:
            if restart_obj:
                obj = []
                t1 = time.time()
                o, norms = self.Demult_Obj_Get(rho, fpm, k, S, f1, f2, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
                obj.append(o)
                t2 = time.time()

                print("="*80)
                print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
                print("="*80)
                self.Save_Params(np.array(norms), progress_dir+\
                    '/norms_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv')
            else:
                pass

        for e in range(epochs):
            t1 = time.time()
            bulbs = bulb_idx
            bulbs_left = num_bulbs
            for s in range(per_epoch):
                # Sample bulbs in array without replacement
                if 2*sample < bulbs.shape[0]:
                    samp = np.random.choice(bulbs_left, sample, replace = False)
                    iter_bulbs = bulbs[samp]
                    bulbs = np.delete(bulbs, samp)
                    bulbs_left -= sample
                else:
                    iter_bulbs = bulbs

                # Adjust sampled bulbs
                rho_new = np.copy(rho)
                rho_new[iter_bulbs] = rho[iter_bulbs] +\
                                    np.random.normal(0, p, iter_bulbs.shape)

                if verbose:
                    print("-"*80)
                    print("Bulbs sampled:" , iter_bulbs+1)
                    print("fp before:", self.Scale_Rho_fp(rho[iter_bulbs],self.f_a(fpm)))
                    print("fp after:", self.Scale_Rho_fp(rho_new[iter_bulbs],self.f_a(fpm)))
                    print("-"*80)

                # Compute objective
                o, norms = self.Demult_Obj_Get(rho, fpm, k, S, f1, f2,\
                                                df, objective, norms, duty_cycle)
                
                if optimizer == 'grad. asc.':
                    # Compute gradient
                    grad = (o-obj[len(obj)-1])/\
                            (rho_new[iter_bulbs]-rho[iter_bulbs]+1e-10)

                    # Gradient Ascent
                    rho[iter_bulbs] = rho_evolution[rho_evolution.shape[0]-1,\
                                                    iter_bulbs] + alpha*grad
                elif optimizer == 'greedy search':
                    if o > obj[len(obj)-1]:
                        rho[iter_bulbs] = rho_new[iter_bulbs]
                    else:
                        pass
                else:
                    raise RuntimeError("That optimizer is not implemented.")

                if verbose:
                    print("-"*80)
                    print("Optimizer adjustment:")
                    print("fp before:\n",\
                        self.Scale_Rho_fp(rho_evolution[rho_evolution.shape[0]-1,\
                                                        iter_bulbs],self.f_a(fpm)))
                    print("fp after:\n", self.Scale_Rho_fp(rho[iter_bulbs],self.f_a(fpm)))
                    print("-"*80)

                # Add to obj and rho tracker
                rho_evolution = np.row_stack([rho_evolution, rho])
                obj.append(o)
                print("Epoch: %3d/%3d | Sample: %3d/%3d | Value: %5e"\
                        %(e+1, epochs, s+1, per_epoch, o))

            t2 = time.time()
            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e"\
                        %(e+1, epochs, t2-t1, o))
            print("="*80)
            
            self.Save_Params(rho_evolution, progress_dir+\
                    '/rho_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv')
            self.Save_Params(np.array(obj), progress_dir+\
                    '/obj_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'%(f1,f2,fpm)+ID+'.csv')

        best_iter = np.argmax(np.array(obj))
        self.Demult_Run_And_Plot(progress_dir, rho_evolution[best_iter,:], fpm, k, S, f1, f2,\
                            fwin = fwin, show = show)
        self.Plot_Obj(progress_dir+'/obj_Demult_%.1f_%.1fGHz_fpm_%.1fGHz'\
                              %(f1,f2,fpm)+ID+'.pdf', np.array(obj))

        return


    def Demult_Obj_Get(self, rho, fpm, k, S, f1, f2, df = 0.25,\
                            objective = 'comp', norms = [], duty_cycle = 0.5):
        """
        Run array and get one objective value evaluation.

        Args:
            See args for Optimize_Demultiplexer()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')
        time.sleep(18/duty_cycle-20)

        if objective == 'comp':
            return Demult_Obj_Comp(freq/10**9, S21, S31, f1, f2, df, norms)
        elif objective == 'dB':
            return Demult_Obj_dB(freq/10**9, S21, S31, f1, f2, df, norms)
        else:
            raise RuntimeError("That objective has not been implemented")


    def Demult_Run_And_Plot(self, save_dir, rho, fpm, k, S, f1, f2,\
                                   fwin = [], show = True):
        """
        Run array and plot transmission spectrumn.

        Args:
            See args for Optimize_Demultiplexer() and Trans_Plot_2Port()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')

        savepath = save_dir+'/Demult_%.1f_%.1fGHz_fpm_%.1fGHz_k%.1f_S%.1f.pdf'\
                        %(f1,f2,fpm,k,S)
        self.Trans_Plot_2Port(savepath, freq/10**9, S21, S31, fpm, k, S,\
                              f = [f1, f2], f_win = fwin, show = show)

        return


    def optimize_waveguide_stochastic(self, epochs, rho, fpm, k, S, f, df = 0.5,\
                               alpha = 0.001, sample = 12, p = 0.01,\
                               objective = 'comp', optimizer = 'grad. asc.',\
                               wu = 10, progress_dir = '.', fwin = [],\
                               duty_cycle = 0.5, show = True,\
                               restart_obj = False, verbose = False,\
                               ID = '', snapback=False):
        """
        Performs an in-situ optimization procedure to produce a waveguide/beam
        steering device that operates at freqeuncy f and directs signal into
        port 2.

        Args:
            epochs: int, Number of epochs (1 epoch = all bulbs adjusted once)
            rho: np.array, Starting parameters
            fpm: float, Max plasma frequency in GHz
            k: float in [0,1], Bulb fit knob
            S: float in [0,1], Bulb fit scale factor
            f: float, operating frequency in GHz
            df: float, bandwidth around operating frequencies in GHz
            alpha: float, learning rate
            sample: int, How many bulbs at a time are modified to compute 
                    gradient wrt subset of parameters
            p: float, std. dev. of noise added to parameters
            objective: str, chooses objective function
            wu: int, # of minutes to warm up the array
        """
        if os.path.isfile(progress_dir+'/rho_Wvg_%.1fGHz_fpm_%.1fGHz'\
                                          %(f, fpm)+ID+'.csv'):
            obj = self.Read_Params(progress_dir+\
                    '/obj_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv').tolist()
            norms = self.Read_Params(progress_dir+\
                    '/norms_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv').tolist()
            rho_evolution = self.Read_Params(progress_dir+\
                    '/rho_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv')
            rho = np.copy(rho_evolution[np.argmax(obj),:])
            print('='*80)
            print('NOTE: Optimizer starting over from sample '+\
                    '%d of previous run'%(np.argmax(obj)+1))
            print('='*80)
            continuation = True
        else:
            rho_evolution = np.zeros((1,rho.shape[0]))
            rho_evolution[0,:] = np.copy(rho)
            continuation = False
        
        num_bulbs = rho.shape[0]
        bulb_idx = np.array(list(range(num_bulbs)))

        if num_bulbs%sample != 0:
            per_epoch = num_bulbs//sample
        else:
            per_epoch = num_bulbs//sample
        
        print("="*80)
        print("Initiating waveguide optimization. You have chosen to run "+\
              str(epochs)+" epochs with a\nsample factor of "+str(sample)+".")
        print("Since there are "+str(num_bulbs)+" bulbs, this means that each epoch"+\
              " will take %.1f minutes, for\na total runtime of"%(per_epoch*38/60)+\
              " about %.1f minutes."%(per_epoch*epochs+wu))
        print("="*80)
        print("\n")
        print("="*80)
        print("Running array warmup.")
        print("="*80)
        self.Config_Warmup(T = wu, ballasts = 'New', duty_cycle = duty_cycle)

        print("\n")
        print("="*80)
        print("Array warm! Beginning optimization.")
        print("="*80)

        if not continuation:
            obj = []
            t1 = time.time()
            o, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
            obj.append(o)
            t2 = time.time()

            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
            print("="*80)
            self.Save_Params(np.array(norms), progress_dir+\
                '/norms_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv')
        else:
            if restart_obj:
                t1 = time.time()
                o, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f, df,\
                                        objective, norms = [],\
                                        duty_cycle = duty_cycle)
                obj.append(o)
                t2 = time.time()

                print("="*80)
                print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(0, epochs,\
                                                                    t2-t1, o))
                print("="*80)
                self.Save_Params(np.array(norms), progress_dir+\
                    '/norms_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv')
            else:
                o = max(obj)

        best_obj_so_far = o
        best_rho_so_far = np.copy(rho)

        for e in range(epochs):
            if snapback and (e > 0) and (e % 5 == 0):
                print("="*80)
                print(f"Epoch {e+1}: Snapping back to best rho (Objective: {best_obj_so_far:.5e}).")
                print("="*80)
                rho = np.copy(best_rho_so_far)
            
            t1 = time.time()
            bulbs = bulb_idx
            bulbs_left = num_bulbs
            for s in range(per_epoch):
                # Sample bulbs in array without replacement
                if 2*sample < bulbs.shape[0]:
                    samp = np.random.choice(bulbs_left, sample, replace = False)
                    iter_bulbs = bulbs[samp]
                    bulbs = np.delete(bulbs, samp)
                    bulbs_left -= sample
                else:
                    iter_bulbs = bulbs

                # Adjust sampled bulbs
                rho_new = np.copy(rho)
                rho_new[iter_bulbs] = rho[iter_bulbs] +\
                                    np.random.normal(0, p, iter_bulbs.shape)

                if verbose:
                    print("-"*80)
                    print("Bulbs sampled:" , iter_bulbs+1)
                    print("fp before:", self.Scale_Rho_fp(rho[iter_bulbs],self.f_a(fpm)))
                    print("fp after:", self.Scale_Rho_fp(rho_new[iter_bulbs],self.f_a(fpm)))
                    print("-"*80)

                # Compute objective
                o, norms = self.Wvg_Obj_Get(rho_new, fpm, k, S, f,\
                                                df, objective, norms, duty_cycle)

                if o > best_obj_so_far:
                    best_obj_so_far = o
                    best_rho_so_far = np.copy(rho_new)
                
                if optimizer == 'grad. asc.':
                    # Compute gradient
                    grad = (o-obj[len(obj)-1])/(rho_new[iter_bulbs]-rho[iter_bulbs]+1e-10) #adding tiny number to make sure not 0

                    # Gradient Ascent
                    rho[iter_bulbs] = rho_evolution[rho_evolution.shape[0]-1,\
                                                iter_bulbs] + alpha*grad
                elif optimizer == 'greedy search':
                    if o > obj[len(obj)-1]:
                        rho[iter_bulbs] = rho_new[iter_bulbs]
                    else:
                        pass
                else:
                    raise RuntimeError("That optimizer is not implemented.")

                if verbose:
                    print("-"*80)
                    print("Optimizer adjustment:")
                    print("fp before:\n",\
                        self.Scale_Rho_fp(rho_evolution[rho_evolution.shape[0]-1,\
                                                        iter_bulbs],self.f_a(fpm)))
                    print("fp after:\n", self.Scale_Rho_fp(rho[iter_bulbs],self.f_a(fpm)))
                    print("-"*80)

                # Add to obj and rho tracker
                rho_evolution = np.row_stack([rho_evolution, rho])
                obj.append(o)
                print("Epoch: %3d/%3d | Sample: %3d/%3d | Value: %5e"\
                        %(e+1, epochs, s+1, per_epoch, o))

            t2 = time.time()
            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e"\
                        %(e+1, epochs, t2-t1, o))
            print("="*80)
            
            # Plot intermediate results after every 5 epochs
            if (e + 1) % 5 == 0 and e < epochs - 1:
                print(f"--- Plotting intermediate results for Epoch {e + 1} ---")
                
                obj_savepath = progress_dir + f'/obj_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}_epoch_{e+1}.pdf'
                self.Plot_Obj(obj_savepath, np.array(obj), show=show)
                
                print("Taking a snapshot measurement of the best state so far...")
                try:
                    self.ArraySet_Rho(best_rho_so_far, self.f_a(fpm), knob=k, scale=S)
                    time.sleep(1)
                    freq_snap, s21_snap, s31_snap = self.Get_S21_S31()
                finally:
                    self.Deactivate_Bulb('all') 
                
                s_param_savepath = progress_dir + f'/Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz_k{k:.1f}_S{S:.1f}_epoch_{e+1}.pdf'
                self.Trans_Plot_2Port(s_param_savepath, freq_snap/1e9, s21_snap, s31_snap, fpm, k, S, f=[f], f_win=fwin, show=show)
                print("--- Intermediate plots saved. Continuing optimization... ---")

            self.Save_Params(rho_evolution, progress_dir+\
                    '/rho_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv')
            self.Save_Params(np.array(obj), progress_dir+\
                    '/obj_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.csv')

        best_iter = np.argmax(np.array(obj))
        self.Wvg_Run_And_Plot(progress_dir, rho_evolution[best_iter,:], fpm, k, S, f,\
                            fwin = fwin, show = show)
        self.Plot_Obj(progress_dir+'/obj_Wvg_%.1fGHz_fpm_%.1fGHz'%(f, fpm)+ID+'.pdf',\
                      np.array(obj))

        return
    
    


    def optimize_waveguide_bayes_old(self, epochs, rho, fpm, k, S, f,
        df=0.5, sample=12, p_range=0.10,
        n_calls=15, n_init=5,
        objective='comp', wu=10, progress_dir='.',
        fwin=[], duty_cycle=0.5, show=True,
        restart_obj=False, verbose=False, ID=''):
        """
        Bayesian optimization version of waveguide/beam-steering in-situ tuning.
        """
        os.makedirs(progress_dir, exist_ok=True)

        # progress paths (match stochastic names)
        rho_path = f"{progress_dir}/rho_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.csv"
        obj_path = f"{progress_dir}/obj_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.csv"
        nrm_path = f"{progress_dir}/norms_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.csv"

        # resume / init
        if os.path.isfile(rho_path):
            obj   = self.Read_Params(obj_path).tolist()
            norms = self.Read_Params(nrm_path).tolist()
            rho_evolution = self.Read_Params(rho_path)
            rho = np.copy(rho_evolution[int(np.argmax(obj)), :])
            print('='*80)
            print('NOTE: Optimizer starting over from sample %d of previous run'
                % (int(np.argmax(obj))+1))
            print('='*80)
        else:
            rho_evolution = np.zeros((1, rho.shape[0]))
            rho_evolution[0, :] = np.copy(rho)
            obj = []
            norms = []

        num_bulbs = rho.shape[0]
        bulbs_all = np.arange(num_bulbs)

        per_epoch = max(1, math.ceil(num_bulbs / max(1, sample))) #delete if i delete samples...?

        print("="*80)
        print("Initiating waveguide (Bayesian) optimization. You have chosen to run "
            f"{epochs} epochs with a\nsample factor of {sample}.")
        print("Since there are " + str(num_bulbs) + " bulbs, this means that each epoch"
            f" will take {per_epoch*38/60:.1f} minutes, for\na total runtime of"
            f" about {per_epoch*epochs + wu:.1f} minutes.")
        print("="*80)
        print("\n")
        print("="*80)
        print("Running array warmup.")
        print("="*80)
        self.Config_Warmup(T=wu, ballasts='New', duty_cycle=duty_cycle)
        print("\n")
        print("="*80)
        print("Array warm! Beginning optimization.")
        print("="*80)

        # initial objective (Epoch 0)
        # initial objective (Epoch 0) — get fixed norms once, then baseline using them
        if (len(obj) == 0) or restart_obj or (len(norms) == 0):
            t1 = time.time()
            # first call just to obtain norms (apples-to-apples scaling)
            _, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f, df, objective, [], duty_cycle)
            # real baseline objective using those fixed norms
            o0, _ = self.Wvg_Obj_Get(rho, fpm, k, S, f, df, objective, norms, duty_cycle)
            obj = [o0]  # reset to a comparable baseline
            t2 = time.time()
            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e"
                % (0, epochs, t2 - t1, o0))
            print("="*80)
            self.Save_Params(np.array(norms), nrm_path)

        force_change_next = False
        
        # main loop
        for e in range(epochs):
            t_epoch_start = time.time()
            
            rho_epoch_start = np.copy(rho)


            bulbs = bulbs_all.copy()
            s_count = 0

            # --- Drain until empty: take chunks of size min(sample, remaining) ---
            while bulbs.size > 0:
                s_count += 1
                # If last sample reverted / made zero change, force a small random nudge this time
                if force_change_next:
                    take = min(sample, bulbs.size)
                    idx = np.random.choice(bulbs.size, size=take, replace=False)
                    block = bulbs[idx]
                    bulbs = np.delete(bulbs, idx)

                    base = np.copy(rho)

                    # guaranteed non-zero move (kept within safe range)
                    forced = 0.05 * np.random.uniform(-p_range, p_range, size=block.size)
                    trial_after = np.copy(base)
                    trial_after[block] = np.clip(base[block] + forced, 0.0, self.f_a(fpm))

                    # commit without revert gate (we *want* a change)
                    rho[block] = trial_after[block]

                    # log/track
                    best_val, _ = self.Wvg_Obj_Get(rho, fpm, k, S, f, df, objective, norms, duty_cycle)
                    if verbose:
                        V_before = self.Rho_to_Bulb(base, self.f_a(fpm), knob=k, scale=S)[:, 0]
                        V_after  = self.Rho_to_Bulb(rho,  self.f_a(fpm), knob=k, scale=S)[:, 0]
                        dV_block = (V_after - V_before)[block]
                        print(f"[epoch {e+1} sample {s_count}] FORCED change; ΔV(V)={np.round(dV_block,3).tolist()}")

                    rho_evolution = np.row_stack([rho_evolution, rho])
                    obj.append(best_val)
                    print("Epoch: %3d/%3d | Sample: %3d/%3d | Value: %5e (forced)"
                        % (e+1, epochs, s_count, per_epoch, best_val))
                    self.Save_Params(rho_evolution, rho_path)
                    self.Save_Params(np.array(obj), obj_path)

                    force_change_next = False
                    continue
                take = min(sample, bulbs.size)
                idx = np.random.choice(bulbs.size, size=take, replace=False)
                block = bulbs[idx]
                bulbs = np.delete(bulbs, idx)

                base = np.copy(rho)

                base_val, _ = self.Wvg_Obj_Get(
                    base, fpm, k, S, f, df, objective, norms, duty_cycle)


                def eval_delta(delta_vec):
                    delta = np.clip(np.asarray(delta_vec, float), -p_range, p_range)
                    trial = np.copy(base)
                    trial[block] = base[block] + delta
                    trial = np.clip(trial, 0.0, self.f_a(fpm))  # keep sane range
                    val, _ = self.Wvg_Obj_Get(
                        trial, fpm, k, S, f, df, objective, norms, duty_cycle)
                    return float(val)

                def skopt_obj(x):
                    return -eval_delta(x)

                from skopt.space import Real
                space = [Real(-p_range, p_range, name=f"d{i}") for i in range(block.size)]
                n_calls_eff = max(int(n_calls), int(n_init))
                n_init_eff  = min(int(n_init), n_calls_eff)

                res = gp_minimize(
                    skopt_obj, space,
                    n_calls=n_calls_eff,
                    n_initial_points=n_init_eff,
                    noise="gaussian",
                    acq_func="EI"
                )

                best_delta = np.array(res.x, float)
                trial_after = np.copy(base)
                trial_after[block] = base[block] + best_delta
                trial_after = np.clip(trial_after, 0.0, self.f_a(fpm))

                # --- 3) REVERT IF WORSE (compare apples-to-apples with same norms) ---
                final_val, _ = self.Wvg_Obj_Get(
                    trial_after, fpm, k, S, f, df, objective, norms, duty_cycle)
                
                if final_val >= base_val - 1e-6:
                    rho[block] = trial_after[block]   # commit
                    best_val = final_val
                else:
                    rho[block] = base[block]          # revert
                    best_val = base_val
                    force_change_next = True
                    
                if verbose:
                    # ΔV list relative to the previous step (before commit)
                    V_before_all = self.Rho_to_Bulb(base, self.f_a(fpm), knob=k, scale=S)[:, 0]
                    V_after_all  = self.Rho_to_Bulb(rho,  self.f_a(fpm), knob=k, scale=S)[:, 0]
                    dV_all = V_after_all - V_before_all
                    print(f"[epoch {e+1} sample {s_count}] ΔV_all(V)={np.round(dV_all, 3).tolist()}")

                if np.max(np.abs(dV_all)) < 1e-3:   # choose tolerance like 1mV
                    force_change_next = True


                rho_evolution = np.row_stack([rho_evolution, rho])
                obj.append(best_val)
                print("Epoch: %3d/%3d | Sample: %3d/%3d | Value: %5e"
                    % (e+1, epochs, s_count, per_epoch, best_val))

                self.Save_Params(rho_evolution, rho_path)
                self.Save_Params(np.array(obj), obj_path)


            # end-of-epoch print
            t_epoch_end = time.time()
            print("="*80)
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e"
                % (e+1, epochs, t_epoch_end - t_epoch_start, obj[-1]))
            print("="*80)
            
            if verbose:
                V0 = self.Rho_to_Bulb(rho_epoch_start, self.f_a(fpm), knob=k, scale=S)[:, 0]
                V1 = self.Rho_to_Bulb(rho,              self.f_a(fpm), knob=k, scale=S)[:, 0]
                dV_epoch = V1 - V0
                print(f"[epoch {e+1}] ΔV_epoch_all(V)={np.round(dV_epoch, 3).tolist()}")


            # Plot every epoch except the final one 
            if e < epochs - 1:
                print(f"--- Plotting intermediate results for Epoch {e + 1} ---")
                obj_savepath = progress_dir + f'/obj_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}_epoch_{e+1}.pdf'
                #self.Plot_Obj(obj_savepath, np.array(obj), show=show)
                self.Plot_Obj(obj_savepath, np.array(obj[per_epoch::per_epoch]), show=show)


                best_iter_so_far = int(np.argmax(np.array(obj)))
                best_rho_for_plot = rho_evolution[best_iter_so_far, :]
                print("Taking a snapshot measurement of the best state so far...")
                try:
                    self.ArraySet_Rho(best_rho_for_plot, self.f_a(fpm), knob=k, scale=S)
                    time.sleep(1)
                    freq_snap, s21_snap, s31_snap = self.Get_S21_S31()
                finally:
                    self.Deactivate_Bulb('all')
                s_param_savepath = progress_dir + f'/Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz_k{k:.1f}_S{S:.1f}_epoch_{e+1}.pdf'
                csv_savepath = s_param_savepath.replace('.pdf', '.csv') #also csv
                np.savetxt(csv_savepath, #also csv
                        np.column_stack([freq_snap/1e9, s21_snap, s31_snap]),
                        delimiter=',', header='freq_GHz,S21_dB,S31_dB', comments='')
                self.Trans_Plot_2Port(
                    s_param_savepath, freq_snap/1e9, s21_snap, s31_snap,
                    fpm, k, S, f=[f], f_win=fwin, show=show
                )
                print("--- Intermediate plots saved. Continuing optimization... ---")

        best_i = int(np.argmax(np.array(obj)))
        self.Wvg_Run_And_Plot(
            progress_dir, rho_evolution[best_i, :],
            fpm, k, S, f, fwin=fwin, show=show
        )
        self.Plot_Obj(
            # f"{progress_dir}/obj_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.pdf",
            # np.array(obj)
            f"{progress_dir}/obj_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.pdf",
            np.array(obj[per_epoch::per_epoch])
        )
        return

    def optimize_waveguide_bayes(self, n_total_calls, rho, fpm, k, S, f,
                             df=0.5, n_initial_points=10,
                             objective='comp', wu=10, progress_dir='.',
                             fwin=[], duty_cycle=0.5, show=True, ID=''):
        import os
        import numpy as np
        from skopt import gp_minimize
        from skopt.space import Real

        os.makedirs(progress_dir, exist_ok=True)

        # Define file paths
        rho_path = f"{progress_dir}/rho_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.csv"
        obj_path = f"{progress_dir}/obj_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.csv"
        nrm_path = f"{progress_dir}/norms_Wvg_{f:.1f}GHz_fpm_{fpm:.1f}GHz{ID}.csv"

        # --- ENSURE NORMS EXIST (compute or load once) ---
        norms = None
        if os.path.isfile(nrm_path):
            print(f"Loading norms from {nrm_path}")
            norms = np.loadtxt(nrm_path, delimiter=',')
            if np.ndim(norms) == 0:
                norms = [norms]
        else:
            print("No existing norms found. Computing initial norms...")
            _, norms = self.Wvg_Obj_Get(rho, fpm, k, S, f, df, objective, [], duty_cycle)
            np.savetxt(nrm_path, np.array(norms), delimiter=',')
            print(f"Saved new norms to {nrm_path}")

        # --- WARM/COLD START DETECTION (KEEPING WARM START) ---
        x0_initial, y0_initial = None, None
        n_calls_remaining = n_total_calls
        is_warm_start = False

        if os.path.isfile(rho_path) and os.path.isfile(obj_path):
            print("="*80)
            print("Found existing logs. Attempting WARM START...")
            loaded_rhos = self.Read_Params(rho_path)
            loaded_objs = self.Read_Params(obj_path)

            if loaded_rhos.ndim == 1: loaded_rhos = np.array([loaded_rhos])
            if loaded_objs.ndim == 0: loaded_objs = np.array([loaded_objs])

            if loaded_rhos.shape[0] == loaded_objs.shape[0]:
                x0_initial = loaded_rhos.tolist()
                y0_initial = (-loaded_objs).tolist()  # negate for minimizer
                is_warm_start = True

                n_done = len(x0_initial)
                n_calls_remaining = n_total_calls - n_done
                if n_calls_remaining <= 0:
                    print(f"Optimization already completed {n_done}/{n_total_calls} calls. Exiting.")
                    return
                print(f"Loaded {n_done} evaluations. Running for {n_calls_remaining} more.")
                print("="*80)
            else:
                print("Warning: Log file mismatch. Starting fresh.")
                x0_initial = [rho.tolist()]
        else:
            print("No logs found. Starting COLD START.")
            x0_initial = [rho.tolist()]

        # If COLD start, truncate/create empty CSVs so the callback appends cleanly
        if not is_warm_start:
            open(rho_path, 'w').close()
            open(obj_path, 'w').close()

        # --- DEFINE OBJECTIVE FUNCTION ---
        def full_objective_function(rho_candidate):
            rho_candidate = np.array(rho_candidate)
            val, _ = self.Wvg_Obj_Get(
                rho_candidate, fpm, k, S, f, df, objective, norms, duty_cycle
            )
            return -val  # negative for minimization

        # --- ALWAYS use the callback (cold AND warm) ---
        callback_handler = save_progress_callback(
            rho_path, obj_path, self, fpm, k, S, f, fwin, show, progress_dir
        )

        # --- RUN THE OPTIMIZER ---
        result = gp_minimize(
            func=full_objective_function,
            dimensions=[Real(0.0, self.f_a(fpm), name=f"rho_{i}") for i in range(rho.shape[0])],
            x0=x0_initial,
            y0=y0_initial,
            n_calls=n_calls_remaining,
            n_initial_points=max(0, n_initial_points - (len(x0_initial) if is_warm_start else 0)),
            noise="gaussian",
            acq_func="EI",
            callback=callback_handler
        )

        # --- FINAL PROCESSING ---
        all_rhos_tested = np.array(result.x_iters)
        all_objs_achieved = -np.array(result.func_vals)
        best_rho_found = all_rhos_tested[np.argmax(all_objs_achieved)]

        print("\nOptimization complete. Best objective found: ", np.max(all_objs_achieved))

        # Final plot of the best state + overall objective PDF (not per-call; callback already saved per-call)
        self.Wvg_Run_And_Plot(progress_dir, best_rho_found, fpm, k, S, f, fwin=fwin, show=show)
        self.Plot_Obj(obj_path.replace('.csv', '.pdf'), all_objs_achieved, show=show)

        return
    
    def Wvg_Obj_Get(self, rho, fpm, k, S, f, df = 0.25,\
                    objective = 'comp', norms = [], duty_cycle = 0.5):
        """
        Run array and get one objective value evaluation.

        Args:
            See args for optimize_waveguide_stochastic()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')
        time.sleep(18/duty_cycle-20)

        if objective == 'comp':
            return Waveguide_Obj_Comp(freq/10**9, S21, S31, f, df, norms)
        elif objective == 'dB':
            return Waveguide_Obj_dB(freq/10**9, S21, S31, f, df, norms)
        elif objective == 'narrow':  # <--- add this
            return Waveguide_Obj_Narrow(freq/1e9, S21, S31, f, df, norms, w_in=1.0, w_oob=0.5)
        else:
            raise RuntimeError("That objective has not been implemented")


    def Wvg_Run_And_Plot(self, save_dir, rho, fpm, k, S, f,\
                                   fwin = [], show = True):
        """
        Run array and plot transmission spectrumn.

        Args:
            See args for optimize_waveguide_stochastic() and Trans_Plot_2Port()
        """
        self.ArraySet_Rho(rho, self.f_a(fpm), knob = k, scale = S)
        time.sleep(1)
        freq, S21, S31 = self.Get_S21_S31()
        self.Deactivate_Bulb('all')
        time.sleep(1)
        self.Deactivate_Bulb('all')

        savepath = save_dir+'/Wvg_%.1fGHz_fpm_%.1fGHz_k%.1f_S%.1f.pdf'\
                        %(f,fpm,k,S)
        self.Trans_Plot_2Port(savepath, freq/10**9, S21, S31, fpm, k, S,\
                              f = [f], f_win = fwin, show = show)
        
        csvpath = savepath.replace('.pdf', '.csv')
        np.savetxt(csvpath,
                np.column_stack([freq/1e9, S21, S31]),
                delimiter=',', header='freq_GHz,S21_dB,S31_dB', comments='')

        return


    def Save_Params(self, rho, savepath):
        """
        Wrapper for np.savetxt
        """
        return np.savetxt(savepath, rho, delimiter=',')


    def Read_Params(self, readpath, iteration = 0):
        """
        Reads optimization parameters from the computational invdes library.

        Args:
            readpath: read path. Must be csv.
        """
        if iteration > 0:
            rho = np.loadtxt(readpath, delimiter=',')
            return rho[iteration-1,:]
        elif iteration == 'last':
            rho = np.loadtxt(readpath, delimiter=',')
            return rho[rho.shape[0]-1,:]
        else:
            return np.loadtxt(readpath, delimiter=",")


    def f_GHz(self, f):
        """
        Returns dimensionalized frequency in GHz

        Args:
            f: frequency in a units
        """
        return f*c/self.a/10**9
    

    def f_a(self, f):
        """
        Returns nondimensionalized frequency in a units

        Args:
            f: frequency in GHz
        """
        return f*10**9/c*self.a


    def Trans_Plot_2Port(self, savepath, freq, S21, S31, fpm, k, S,\
                         f = [], f_win = [], show = True):
        """
        Creates plot of transmission spectrum for 2-port measurement

        Args:
            savepath: str
            freq: np.array, frequency in GHz
            S21: np.array
            S31: np.array
            fpm: float, max frequency in GHz
            f: list of floats, operating frequencies in GHz
        """
        fig, ax = plt.subplots(1,1,figsize=(9,6))

        ax.set_xlabel('Frequency (GHz)', fontsize = 30)
        ax.set_ylabel('$S_{31}$ and $S_{21}$ (dB)', fontsize = 30)
        for i in range(10):
            ax.axhline(y=-10*(i+1), color='grey', label='_nolegend_',\
                       linewidth = 1)
        ax.set_title('k = %.1f, S = %.1f, $f_{p,max}$ = %.1f GHz'%(k,S,fpm),\
                     fontsize = 30)
        ax.tick_params(labelsize = 27)
        ax.plot(freq, S21, linewidth = 5)
        ax.plot(freq, S31, linewidth = 5)
        for freq in f:
            ax.axvline(x=freq, color='k', linestyle='--')
        if len(f_win) > 0:
            ax.set_xlim(f_win)
        ax.set_ylim([-80,-10])
        ax.legend(['$S_{21}$','$S_{31}$'], bbox_to_anchor=[1.15, 0.5], loc = 'center', ncol = 1, fontsize = 27)

        plt.savefig(savepath, dpi=1500, bbox_inches='tight')
        if show:
            plt.show()

        return


    def Plot_Obj(self, savepath, obj, show = True):
        """
        Creates plot of objective evolution throughout optimization

        Args:
            savepath: str
            obj: np.array, objective function values
        """
        fig, ax = plt.subplots(1,1,figsize=(9,6))

        ax.set_xlabel('Samples', fontsize = 30)
        ax.set_ylabel('Objective', fontsize = 30)
        ax.tick_params(labelsize = 27)
        ax.plot(obj, linewidth = 5)

        plt.savefig(savepath, dpi=1500, bbox_inches='tight')
        if show:
            plt.show()

        return
