Folder for samples. I tweaked it to be a symlink to a shared memory folder `/dev/shm/traindata`. This way file IO is faster, but you'll lose all the samples data if you forget to save them to a real folder.

If you are restarting the training from scratch, remove all `*.dat` files
