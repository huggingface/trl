local_dir=${2:-"./"}

CMD="hf download nvidia/jet-dev-share --include "$1" --local-dir "$local_dir" --repo-type dataset"

echo $CMD
$CMD