import zmq 
import signal 

import sys 

import multiprocessing as mp 

from typing import List, Dict, Any, Generator, Type 

from contextlib import suppress

from imagetopic.schema.embedding import TaskResponse
from imagetopic.embedding.strategy import ABCStrategy
from imagetopic.log import logger 

class GENEmbedding:

    def __init__(self, strategy_cls:Type[ABCStrategy], strategy_kwargs:Dict[str, Any]) -> None:
        self.strategy_cls = strategy_cls
        self.strategy_kwargs = strategy_kwargs
        assert issubclass(self.strategy_cls, ABCStrategy)
        super().__init__()

    def _consume_task(self, task:str, strategy:ABCStrategy) -> TaskResponse:
        try:
            content = strategy(task=task)
            return TaskResponse(status=True, content=content, error_message=None)
        except Exception as e:
            error_message = str(e)
            return TaskResponse(status=False, content=None, error_message=error_message)

    def _worker(self, parition_of_tasks:List[str], port:int):
        
        try:
            strategy = self.strategy_cls(**self.strategy_kwargs)
        except Exception as e:
            logger.error(e)
            sys.exit(1)  # strategy builder error

        
        signal.signal(
            signal.SIGTERM,
            lambda signal_n, frame: signal.raise_signal(signal.SIGINT)
        )

        ctx = zmq.Context()
        socket:zmq.Socket = ctx.socket(zmq.PUSH)
        socket.connect(f'tcp://localhost:{port}')

        nb_tasks:int = len(parition_of_tasks)
        idx = 0
        while True:
            try:
                if idx == nb_tasks:
                    break 
                current_task = parition_of_tasks[idx]
                task_response = self._consume_task(task=current_task, strategy=strategy)
                socket.send_pyobj(task_response)
                idx = idx + 1
            except KeyboardInterrupt:
                break 
            except Exception as e:
                logger.error(e)
                break 

        with suppress(Exception):
            del strategy
            
        socket.close(linger=0)
        ctx.term()
    
    def submit_tasks(self, tasks:List[str], nb_workers:int, port:int) -> Generator[TaskResponse, None, None]:
        ctx = zmq.Context()
        socket:zmq.Socket = ctx.socket(zmq.PULL)
        socket.bind(f'tcp://*:{port}')

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        nb_tasks = len(tasks)

        processes:List[mp.Process] = []
        batch_size = max(1, int(nb_tasks / nb_workers))
        for idx in range(0, nb_tasks, batch_size):
            partition_of_tasks = tasks[idx:idx+batch_size]
            process_ = mp.Process(target=self._worker, args=[partition_of_tasks, port])
            processes.append(process_)
            processes[-1].start()
        
        interrupted:bool = False 
        
        counter = 0
        while True:
            try:
                if counter == nb_tasks:
                    break 

                socket_states_hmap:Dict[zmq.Socket, int] = dict(poller.poll(timeout=100))
                incoming_data_flag = socket_states_hmap.get(socket, None)

                if incoming_data_flag == zmq.POLLIN:
                    task_response:TaskResponse = socket.recv_pyobj()        
                    yield task_response
                    counter = counter + 1 

                exitcodes = [ process_.exitcode for process_ in processes if process_.exitcode is not None ]
                if len(exitcodes) == 0:
                    continue

                failed_processes = [ code != 0 for code in exitcodes ]
                if any(failed_processes):
                    logger.warning('a failure was detected from one of the workers')
                    logger.warning('manager system will go down')
                    break 
                
                succeeded_processes = [ code == 0 for code in exitcodes ]
                if all(succeeded_processes):
                    break 

            except KeyboardInterrupt:
                interrupted = True 
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        for process_ in processes:
            if process_.exitcode is not None:
                if not interrupted:  # do not send the SIGTERM if SIGINT was raised 
                    process_.terminate()  # send SIGTERM => raise SIGINT inside process_ loop
                process_.join()  # wait process_ to quit its loop 
            
        poller.unregister(socket)
        socket.close(linger=0)
        ctx.term()

        logger.info('manager resources released')