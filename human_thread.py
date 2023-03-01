import threading
from collections import defaultdict

import numpy as np

#from utils import getch
import getch
from getch import getch#, pause

class FeedbackCollector(threading.Thread):
    '''
    Thread subclass for collecting user key presses from shell
    '''
    def __init__(self, min_feed=-1.0, max_feed=1.0, debug=False):
        threading.Thread.__init__(self)
        self.human_feedback = 0
        self.feedback_lock = threading.Lock()
        self.min_feed = min_feed
        self.max_feed = max_feed
        self.debug = debug
        self.shutdown = False
        self.feedback_histogram = defaultdict(int)

    def run(self):
        signal = 1.
        while not self.shutdown:
            if self.debug:
                print ('Entering feedback collection loop')
            key = getch()
            self.feedback_lock.acquire(True)
            if key == 'g':
                if self.debug:
                    print ('Registered positive human feedback')
                self.human_feedback += signal
                # self.feedback_histogram[key] += 1
            elif key == 'f':
                if self.debug:
                    print ('Registered negative human feedback')
                self.human_feedback -= signal
                # self.feedback_histogram[key] += 1
            elif key == 'q':
                print ('Feedback collector thread signaled to shutdown...')
                self.feedback_lock.release()
                break
                # pause_exit(status=0, message='Press any key to exit.')
            self.human_feedback = np.maximum(self.min_feed, np.minimum(self.max_feed, self.human_feedback))
            self.feedback_lock.release()
            
    def poll(self):
        ret = 0.0
        self.feedback_lock.acquire(True)
        ret += self.human_feedback
        if ret != 0.0:
            key = 'g' if ret > 0.0 else 'f'
            self.feedback_histogram[key] += 1
        if self.debug:
            print ('Polled human feedback signal: {0}'.format(ret))
        self.human_feedback = 0.0
        self.feedback_lock.release()
        return ret

    def print_feedback_stats(self):
        print ('Total feedback signals: {0} | Total positive signals: {1} | Total negative signals: {2}'\
            .format(sum(self.feedback_histogram.values()), self.feedback_histogram['g'], self.feedback_histogram['f']))

    def get_pos_count(self):
        return self.feedback_histogram['g']

    def get_neg_count(self):
        return self.feedback_histogram['f']

    def set_shutdown(self):
        print ('Feedback collector thread signaled to shutdown...')
        self.shutdown = True