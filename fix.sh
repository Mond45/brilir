# create a script that read from file in the following format:
# file:text
# the text is to be prepended to the file at `WORKING_DIR/$file`
# make WORKING_DIR read from argv 1

#!/usr/bin/env bash
WORKING_DIR="$1"
while IFS= read -r line; do
    FILE_PATH="${WORKING_DIR}/$(echo "$line" | cut -d: -f1)"

    echo "Prepending to $FILE_PATH"

    TEXT_TO_PREPEND="$(echo "$line" | cut -d: -f2-)"
    
    # Create the directory if it doesn't exist
    mkdir -p "$(dirname "$FILE_PATH")"
    
    # only prepend if the file exists
    if [ ! -f "$FILE_PATH" ]; then
        echo "File $FILE_PATH does not exist. Skipping."
        continue
    fi

    # Prepend the text to the file
    {
        echo "$TEXT_TO_PREPEND"
        cat "$FILE_PATH"
    } > "${FILE_PATH}.tmp" && mv "${FILE_PATH}.tmp" "$FILE_PATH"
done < args.txt