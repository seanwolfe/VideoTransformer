#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <arg1>"
    exit 1
fi

# Assign arguments to variables
sync_to_remote=$1

# Files
root=~/Documents/sean/AsteroidImageGenerator/
dest=wolfesea@niagara.scinet.utoronto.ca:/scratch/e/emamirez/wolfesea/AsteroidImageGenerator/
file4=to_sync
files=("$file4")


# Print the argument
echo "Argument: $sync_to_remote"

# Construct the list of files for scp
file_list=""
for file in "${files[@]}"; do
  file_list+="$root/$file "
done

# Trim the trailing space
file_list=$(echo $file_list | sed 's/ $//')

# Sync files
if [ "$sync_to_remote" == 'yes' ]; then
  echo "Syncing from local to remote"
  scp -r $file_list $dest
elif [ "$sync_to_remote" == 'no' ]; then
  echo "Syncing from remote to local"
  remote_files=""
  for file in "${files[@]}"; do
    remote_files+="$dest/$file"
  done
  # Trim the trailing space
  remote_files=$(echo $remote_files | sed 's/ $//')
  scp -r $remote_files $root
else
  echo "Error: specify yes or no"
fi
