CMD="rclone copy \
    --checkers=8 \
    --transfers=8 \
    --multi-thread-streams=8 \
    --buffer-size=128M \
    --copy-links \
    --progress \
    pbss-nvr-elm-han:$1 $2
"

echo $CMD
$CMD