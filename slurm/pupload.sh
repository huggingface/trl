CMD="rclone copy \
    --checkers=8 \
    --transfers=8 \
    --multi-thread-streams=8 \
    --buffer-size=128M \
    --copy-links \
    --progress \
    $1 pbss-nvr-elm-han:$2
"

echo $CMD
$CMD