#!/bin/sh

# Example usage:
#
# src/models/hyperparams_train.sh bio_unet -op 'adam sgd' -lr '1e-2 1e-3 1e-4' -bs '4 8' -l 'dice bce' -nc '1 3 5' -d segmentation_496-2d --tpu 8 --sync_request _

is_gcloud() {
    hostname | grep -q ^instance-
    return $?
}

has_tripledash=no
for arg
do
    [ "$arg" = --- ] && { has_tripledash=yes; break; }
done
[ "$has_tripledash" = no ] && {
    # NOTE: this is the "entry point" of the script

    # Setup
    is_gcloud && {
        . /usr/local/share/tpu-stuff
        tpu_start
        sleep 2m # we need this sleep so that the TPU is available
    }

    # Run
    "$0" "$@" ---
    exitcode=$?

    # Cleanup
    is_gcloud && {
        tpu_kill
        sleep 5m
        [ -z "$DIRSYNCC_REQ_DIR" ] || sleep 5m # wait for dirsyncd if variable is set
        sudo sync; sleep 2; sudo shutdown -h now
    }

    exit $exitcode
}

PYTHON=python3

model_name="$1"
flag="$2"
values="$3"

[ "$flag" != --- ] && shift 3

for value in $values
do
    if [ "$flag" = '---' ]
    then
        shift 2
        "$PYTHON" src/models/train.py "$model_name" "$@"
    else
        if [ "$value" = "_" ]
        then
            "$0" "$model_name" "$@" "$flag" || exit $?
        else
            "$0" "$model_name" "$@" "$flag" "$value" || exit $?
        fi
    fi
done
