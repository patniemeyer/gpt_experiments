
#echo "['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '⬛', '⬛', '⬛', '🟥', '⬛', '⬛', '⬛', '\n', '⬛', '⬛', '🟥', '⬛', '⬛', '⬛', '⬛', '\n', '🟥', '⬛', '⬛', '⬛', '⬛', '🟥', '⬛', '\n', '⬛', '⬛', '⬛', '⬛', '⬛', '⬛', '🟥', '\n', '⬛', '⬛', '⬛', '⬛', '🟥', '⬛', '⬛', '\n', '(', '5', ',', '7', ')', '{', '🟥', ':', '7', '}', '.']" | \
tr -d "\'\.\[\]" | \
sed 's/, //g' | sed 's/\\n/\n/g'

