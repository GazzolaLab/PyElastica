"""

#!/bin/bash

#SBATCH -p shared                        # Partition name
#SBATCH -J job_name                         # Job name
(https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E)
#SBATCH -o %x_%j.out                        # Name of stdout output file
#SBATCH -e %x_%j.err                        # Name of stderr error file
#SBATCH -N 1                                # Number of nodes
#SBATCH --ntasks-per-node=32                # cores to use per node
#SBATCH --mem=64G
#SBATCH -t 48:00:00                         # Time limit (hh:mm:ss)
#SBATCH --mail-user=NetID@illinois.edu      # Email notification
#SBATCH --mail-type=ALL                     # Notify on state change (BEGIN/END/FAIL/ALL)
#SBATCH --account=mcb200029p

module load anaconda3
date                                        # Print date

echo Job name: $SLURM_JOB_NAME              # Print job name
echo Execution dir: $SLURM_SUBMIT_DIR       # Print submit directory
echo Number of processes: $SLURM_NTASKS     # Print number of processes

source activate (env_name)
conda env list

export OMP_NUM_THREADS=32
$PROJECT/.conda/envs/(env_name)/bin/python -u (program_name)

"""
from typing import Optional, Literal
import os

ALLOWED_CLUSTERS = Literal["stampede", "bridges", "expanse"]


def create_submit_file(
    program_name: str,
    environment_name: str,
    cluster_name: ALLOWED_CLUSTERS,
    output_file_name: Optional[str] = None,
    error_file_name: Optional[str] = None,
    partition: str = "shared",
    num_nodes: int = 1,
    num_threads: int = 4,
    memory: int = 64,
    time: str = "48:00:00",
    verbose: bool = False,
    mail_user: Optional[str] = None,
    mail_type: Optional[str] = None,
    other_cli_arguments: str = "",
) -> None:

    cluster_info_dict = CLUSTER_MAP[cluster_name]
    filename = "submit_" + program_name.replace(".py", ".sh")
    f = open(filename, "w")
    f.writelines(
        [
            "#!/bin/bash\n",
            "\n",
            f"#SBATCH -p {cluster_info_dict.get(partition)}\n",
            f"#SBATCH -J {program_name.replace('.py', '')}\n",
            f"#SBATCH -N {num_nodes}\n",
            f"#SBATCH -t {time}\n",
            f"#SBATCH --ntasks-per-node={num_threads}\n",
            f"#SBATCH --account={cluster_info_dict.get('account')}\n",
        ]
    )
    # cannot set memory for stampede
    if cluster_name != "stampede":
        f.writelines(
            [
                f"#SBATCH --mem={memory}G\n",
            ]
        )
    if not output_file_name:
        output_file_name = "%x_%j.out"
    if not error_file_name:
        error_file_name = "%x_%j.err"

    f.write(f"#SBATCH -o {output_file_name}\n")
    f.write(f"#SBATCH -e {error_file_name}\n")

    if mail_user:
        f.write(f"#SBATCH --mail-user={mail_user}\n")
        if cluster_name == "stampede":
            print("Set mail user as the complete email-id for stampede!")
        if not mail_type:
            mail_type = "ALL"
        f.write(f"#SBATCH --mail-type={mail_type}\n")

    if verbose:
        f.write("#SBATCH -v\n")

    f.writelines(["\n"])
    # stampede doesn't have conda
    if cluster_name != "stampede":
        f.writelines(["module load anaconda3\n"])
    f.writelines(
        [
            "date\n",
            "\n",
            "echo Job name: $SLURM_JOB_NAME\n",
            "echo Execution dir: $SLURM_SUBMIT_DIR\n",
            "echo Number of processes: $SLURM_NTASKS\n",
            "\n",
            f"source activate {environment_name}\n",
        ]
    )
    # fix stampede issue in pythonpath variable
    if cluster_name == "stampede":
        f.write("unset PYTHONPATH\n")
    f.writelines(
        [
            f"export OMP_NUM_THREADS={num_threads}\n",
            f"python -u {program_name}  {other_cli_arguments}\n",
            "\n",
        ]
    )

    f.close()


if __name__ == "__main__":
    EXPANSE_INFO_DICT = {"account": "uic409", "shared": "shared", "compute": "compute"}
    BRIDGES_INFO_DICT = {
        "account": "mcb200029p",
        "shared": "RM-shared",
        "compute": "RM",
    }
    STAMPEDE_INFO_DICT = {
        "account": "TG-MCB190004",
        "compute": "skx-normal",
    }
    CLUSTER_MAP = {
        "expanse": EXPANSE_INFO_DICT,
        "bridges": BRIDGES_INFO_DICT,
        "stampede": STAMPEDE_INFO_DICT,
    }

    force_mags = [0.0]  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    test_name = "Isobaric"
    muscle_name = "Samuel_supercoil_stl"

    for i in range(len(force_mags)):
        force_mag = force_mags[i]
        folder_name = (
            "passive_force_"
            + test_name
            + "_"
            + muscle_name
            + "_force_mag_{0:2.1f}".format(force_mag)
            + "/"
        )
        os.makedirs(folder_name, exist_ok=True)
        PROGRAM_NAME = "single_muscle_simulator_cluster.py"
        ENVIRONMENT_NAME = "pyelastica-dev-muscle"
        PARTITION = "shared"
        TIME = "08:00:00"
        NUM_THREADS = 4
        MEMORY = 4
        MAIL_USER = "aia@illinois.edu"
        CLUSTER_NAME: ALLOWED_CLUSTERS = "bridges"
        other_cli_arguments = "--test_type " + test_name
        other_cli_arguments += "--muscle_class " + muscle_name
        other_cli_arguments += "--force_mag " + str(force_mag)

        create_submit_file(
            program_name=PROGRAM_NAME,
            environment_name=ENVIRONMENT_NAME,
            cluster_name=CLUSTER_NAME,
            time=TIME,
            memory=MEMORY,
            partition=PARTITION,
            num_threads=NUM_THREADS,
            mail_user=MAIL_USER,
            other_cli_arguments=other_cli_arguments,
        )

        os.system("cp *.py *.npz  " + folder_name)
        os.system("mv *.sh " + folder_name)

        os.chdir(folder_name)

        os.system("pwd")

        os.system("sbatch submit_*")

        os.chdir("../")
