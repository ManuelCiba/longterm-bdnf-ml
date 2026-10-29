import os
import settings
import subprocess
import sys
env = os.environ.copy()

if 0:
    DATA_SET = ["ESTRANGE_DATA3_SHAM_HIP_VS_COR", 
                "ESTRANGE_DATA3_XRAY_COR_VS_SHAM",
                "ESTRANGE_DATA3_XRAY_HIP_VS_SHAM",
                "ESTRANGE_DATA3_XRAY_COR_VS_HIP"]

if 1:
    DATA_SET = ["PsyNet_20260805_DMT_VS_LSD", 
                "PsyNet_20260805_BDNF_VS_LSD", 
                "PsyNet_20260805_BDNF_VS_DMT",
                "PsyNet_20260805_LSD",
                "PsyNet_20260805_DMT",
                "PsyNet_20260805_BDNF"]

if 0:
    DATA_SET = ["PsyNet_20260805_DMT_VS_LSD", 
                "PsyNet_20260805_LSD",
                "PsyNet_20260805_DMT",
                "PsyNet_20260805_BDNF",
                "PsyNet_20260820_DMT_24h",
                "ESTRANGE_DATA3_SHAM_HIP_VS_COR", 
                "ESTRANGE_DATA3_XRAY_COR_VS_SHAM",
                "ESTRANGE_DATA3_XRAY_HIP_VS_SHAM",
                "ESTRANGE_DATA3_XRAY_COR_VS_HIP"]


if __name__ == '__main__':

    for DATA in DATA_SET:

        env["DATA_tmp"] = DATA
        subprocess.run(
            [sys.executable, "main.py"],
            env=env,
            check=True
        )

    