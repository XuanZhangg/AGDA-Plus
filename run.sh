nohup python -u main_DRO.py > ./outputLogs/DRO.log 2>&1 &
nohup python -u main_Q.py > ./outputLogs/Q.log 2>&1 &
nohup python -u main_sinQ.py > ./outputLogs/sinQ.log 2>&1 &

nohup python -u main_DRO.py > ./outputLogs/DRO.log 2>&1 &
nohup python -u plot_Q.py > ./outputLogs/Q.log 2>&1 &
nohup python -u main_sinQ.py > ./outputLogs/sinQ.log 2>&1 &

