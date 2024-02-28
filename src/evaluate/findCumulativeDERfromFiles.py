import os
import re
import subprocess
import math
import argparse
import logging
from tqdm import tqdm

class DERCalculator:
    def __init__(self, ref_rttm_folder_path, sys_rttm_folder_path, resultDERPath):
        self.ref_rttm_folder_path = ref_rttm_folder_path
        self.sys_rttm_folder_path = sys_rttm_folder_path
        self.resultDERPath = resultDERPath
        self.der_metric_txt_name = "der_metric"
        self.derScriptPath = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/src/evaluate/findDER.py"
        self.batch_size = 128
        self.root = "/nlsasfs/home/nltm-st/sujitk/yash-mtp/evaluationResults"
        os.makedirs(self.resultDERPath,exist_ok=True)

    def numeric_part(self, filename):
        return int(''.join(filter(str.isdigit, filename)))

    def findCumulativeDER(self):
        der_files = [os.path.join(self.resultDERPath, filename) for filename in os.listdir(self.resultDERPath) if filename.endswith(".txt")]
        total_der = 0.0
        total_files = 0
        for der_file in der_files:
            with open(der_file, 'r') as f_in:
                der_content = f_in.read()
                overall_match = re.search(r'\*\*\* OVERALL \*\*\*.*?(\d+\.\d+)', der_content, re.DOTALL)
                if overall_match:
                    der = float(overall_match.group(1))
                    total_der += der
                    total_files += 1

        if total_files > 0:
            avg_der = total_der / total_files*1.0
            print(f"The average DER is: {avg_der}")
            return avg_der
        else:
            logging.error("No DER files found or unable to calculate average DER.")


    def run_command(self, command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return None

    def calculateDER(self):
        sys_rttm_files = [os.path.join(self.sys_rttm_folder_path,file) for file in os.listdir(self.sys_rttm_folder_path) if file.endswith(".rttm")]
        sys_rttm = sorted(sys_rttm_files, key=self.numeric_part)
        logging.info(f"System generated rttm files: \n{sys_rttm}")
        
        ref_rttm_files = [os.path.join(self.ref_rttm_folder_path,file) for file in os.listdir(self.ref_rttm_folder_path) if file.endswith(".rttm")]
        ref_rttm = sorted(ref_rttm_files, key=self.numeric_part)
        logging.info(f"The reference/ground truth rttm file: \n{ref_rttm}")
        total_batches = math.ceil(float(len(ref_rttm) * 1.0) / float(self.batch_size))
        start_idx = 0
        
        for batches in tqdm(range(total_batches)):
            end_idx = min(start_idx + self.batch_size, len(ref_rttm))
            temp_sys_rttm = sys_rttm[start_idx:end_idx]
            temp_ref_rttm = ref_rttm[start_idx:end_idx]
            temp_sys_rttm = " ".join(temp_sys_rttm)
            temp_ref_rttm = " ".join(temp_ref_rttm)
            command = f'python {self.derScriptPath} -r {temp_ref_rttm} -s {temp_sys_rttm}'
            output = self.run_command(command)
            der_filename = os.path.join(self.resultDERPath, f"{self.der_metric_txt_name}-{batches}.txt")
            
            with open(der_filename, 'w') as f_out:
                f_out.write(output)
            
            if end_idx == len(ref_rttm):
                break
            start_idx = end_idx + 1
        
        der = self.findCumulativeDER()
        logging.info("Evaluation Completed Successfully!")
        logging.info(f"Please check {self.resultDERPath} for system generated rttm.")


def main():
    parser = argparse.ArgumentParser(description="DER Calculator")
    parser.add_argument("--ref_rttm_folder_path", type=str, required=True, help="Path to the reference RTTM folder")
    parser.add_argument("--sys_rttm_folder_path", type=str, required=True, help="Path to the system RTTM folder")
    parser.add_argument("--out", type=str, required=True, help="Path to the result DER folder")

    args = parser.parse_args()

    calculator = DERCalculator(args.ref_rttm_folder_path, args.sys_rttm_folder_path, args.out)
    calculator.calculateDER()


if __name__ == "__main__":
    main()
