from keras.callbacks import TensorBoard
import time


class TensorBoardColab(TensorBoard):
    def __init__(self, log_dir='./logs', *args, **kwargs):
        port = 6006
        startup_waiting_time = 8
        get_ipython().system_raw('npm i -s -q --unsafe-perm -g ngrok')
        # sudo npm i -s -q --unsafe-perm -g ngrok

        setup_passed = False
        retry_count = 0
        sleep_time = startup_waiting_time / 3.0
        while not setup_passed:
            get_ipython().system_raw('kill -9 $(sudo lsof -t -i:%d)' % port)
            get_ipython().system_raw('rm -Rf ' + log_dir)
            print('Wait for %d seconds...' % startup_waiting_time)
            time.sleep(sleep_time)
            get_ipython().system_raw(
                'tensorboard --logdir %s --host 0.0.0.0 --port %d &' % (
                log_dir, port))
            time.sleep(sleep_time)
            get_ipython().system_raw('ngrok http %d &' % port)
            time.sleep(sleep_time)
            try:
                tensorboard_link = get_ipython().getoutput(
                    'curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin))"')[
                    0]
                tensorboard_link = eval(tensorboard_link)['tunnels'][0]['public_url']
                setup_passed = True
            except:
                setup_passed = False
                retry_count += 1
                print('Initialization failed, retry again (%d)' % retry_count)
                print('\n')

        print("TensorBoard link:")
        print(tensorboard_link)
        super().__init__(log_dir, *args, **kwargs)
