You can specify a Quality of Service (QoS) for each job submitted on the cluster. The associated has two effects on the job:

- **Priority**: depending on the QoS, your job can be more or less prioritized in the SLURM queue.
- **Preemption**: depending on the QoS, your job can preempt other running jobs if insufficient resources are left on the cluster.

| Qos Name    | Priority | Can preempt?    |
| ----------- | -------- | --------------- |
| prod        | ++++     | high,normal,low |
| high,system | +++      | low             |
| normal      | ++       | low             |
| low         | +        | -               |

You can view the latest preemptible settings by running: `sacctmgr show qos`

## Preemption

Job preemption occurs when a job with a QoS higher than the current job cannot be scheduled on the cluster due to a lack of resources. In that case, SLURM will elect job(s) to preempt and send them a `SIGTERM` signal.

## Run a job with QoS

<aside>
❗ When submitting your job with QoS other than `low`, you may end up cancelling some of your other team mates jobs. Use QoS `high` sparingly for jobs that really cannot afford to be preempted.

</aside>

Use the `--qos=<qos_name>` parameter to define the QoS of your job. If omitted, your job ends up in the `normal` QoS.

```bash
# submit a job using the "high" QoS
$ srun --partition=production-cluster \
  --qos=high \
  --time=1-0 \
  --cpus-per-task=1 \
  --mem-per-cpu=1G \
  --pty bash
```

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --nodes=4
# Set the QoS
#SBATCH --qos=high
# set 24h for job wall time limit
#SBATCH --time=1-00:00:00
# activate the requeue option
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/hugo_larcher/logs/my-job/%x-%j-train.out
```

## Has my job been preempted?

You can check the status of your job using:

```bash
$ sacct --jobs=514837  --format=User,JobID,state%30,time,elapsed,AveCPU,MaxRss,ReqMem,nnodes,ncpus,ntasks,NodeList
     User JobID                                 State  Timelimit    Elapsed     AveCPU     MaxRSS     ReqMem   NNodes      NCPUS   NTasks        NodeList
--------- ------------ ------------------------------ ---------- ---------- ---------- ---------- ---------- -------- ---------- -------- ---------------
   ubuntu 514837                            PREEMPTED  UNLIMITED   01:48:24                               1G        1         96           ip-26-0-150-12
          514837.exte+                      COMPLETED              01:48:24   00:00:00          0                   1         96        1  ip-26-0-150-12
          514837.0                          COMPLETED              01:48:23   00:00:00      3056K                   1         96       96  ip-26-0-150-12
```

DRAFT:

Number of `--qos=prod` nodes = 64

Number of `--qos=dev` nodes = 8

Number of `--qos=high,normal` nodes = 96 - 8 = 88

Number of `--qos=high,normal,low` nodes = 96

| Qos Name | Priority | Can preempt?               |
| -------- | -------- | -------------------------- |
| prod     | ++++     | high,normal,low (64 nodes) |
| dev      | ++++     | high,normal,low (8 nodes)  |
| high     | +++      | low (88 nodes)             |
| normal   | ++       | low (88 nodes)             |
| low      | +        | -                          |
