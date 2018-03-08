make ARGS="6 toggle" test

#make ARGS="<0..6> #processing devices 
#	    <steal, shuffle, share, mapreduce>  
#	    <idle, victim, gateway>
#	    <(n n)>
#	    <0..6>" #data resource devices
#	    test
#make ARGS="6 steal gateway 3 3 1" test
#make ARGS="0 steal idle 3 3 1" test

#source config.sh
#device_id="6"
#device_type="gateway"
#ARGUMENTS=$device_id" "$mode" "$device_type" "$dim_and_data_cli
#echo $ARGUMENTS
#make ARGS="$ARGUMENTS" test
