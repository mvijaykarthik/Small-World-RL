ITERS=2
ENSEMBLES=2
EPOCHS=400

DD="agv-complex"
tmp_prefix="rc1"

# Make the directory
if [ ! -e $DD ]; then mkdir $DD; fi;

# At 16x16, total number of options is 2^8 = 256; Run at 100, 200
cmplx=1

#for n in 100 200; do
for n in 200 400 600 800 1000; do
    scheme="small-world"
    # Run for a bunch of 'r'
    for r in 0.75 1.0 1.5 2.0 3.0 4.0; do
        echo "Running $scheme(r=$r) with $n options..."
        #PYTHONOPTIMIZE=3 python2 ./main.py $ITERS $ENSEMBLES $EPOCHS "MacroQ" "RoomsOptions:../domains/rooms-complex/rooms$cmplx.txt:1:$scheme:$n:$r" $tmp_prefix
        PYTHONOPTIMIZE=3 python ./main.py $ITERS $ENSEMBLES $EPOCHS "MacroQ" "AGVOptions:../domains/agv2.txt:$scheme:$n:$r" $tmp_prefix
        mv "$tmp_prefix-return.dat" $DD/$cmplx-$n-$r.return
    done;
done;
