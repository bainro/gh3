import threading
import time
import inspect

class Thread(threading.Thread):
    def __init__(self, t, *args):
        threading.Thread.__init__(self, target=t, args=args)
        # start method inherited from parent class
        self.start()

count = 0
lock = threading.Lock()

def incre():
    global count
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    print("Inside %s()" % caller)
    print("Acquiring lock")
    with lock:
        print("Lock Acquired")
        count += 1  
        time.sleep(2)  

def bye():
    while count < 2:
        incre()

def hello_there():
    while count < 2:
        incre()

def main():    
    hello = Thread(hello_there)
    goodbye = Thread(bye)

if __name__ == '__main__':
    import threading, queue

    q = queue.Queue()

    def worker():
        while True:
            item = q.get()
            print(f'Working on {item}')
            print(f'Finished {item}')
            q.task_done()

    # turn-on the worker thread
    threading.Thread(target=worker, daemon=True).start()

    # send thirty task requests to the worker
    for item in range(30):
        q.put(item)
    print('All task requests sent\n', end='')

    # block until all tasks are done
    q.join()
    print('All work completed')