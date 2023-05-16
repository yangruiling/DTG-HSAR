import numpy as np
import time
import sys

class outputer(object):

    def __init__(self):
        self.start_time = time.time()     #开始时间
        self.all_results = []      #训练集所有结果
        self.result_all_test_data = []    #测试集所有结果
        self.sequence_times = 0

    def init(self):
        self.start_time = time.time()
        self.all_results = []
        self.result_all_test_data = []
        self.sequence_times = 0
    
    def append(self, sequence_results = None, result_whole_test = None):
        if not isinstance(sequence_results, type(None) ):
            self.all_results.append(sequence_results)
        if not isinstance(result_whole_test, type(None)):
            self.result_all_test_data.append(result_whole_test)
        self.sequence_times += 1.0
    
    def output(self):
        self.end_time = time.time()
        print ("-------------------------------------------------")
        avg_result_all_test = np.average(self.result_all_test_data, 0)
        for result_whole_test in self.result_all_test_data:
            self.print_list(result_whole_test)
        print ("-------------------------------------------------")
        self.print_list(avg_result_all_test)
        print ("-------------------------------------------------")
        self.print_avg_results(self.all_results)
        print ("-------------------------------------------------")
        elapsed_time = (self.end_time - self.start_time) / self.sequence_times
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def print_avg_results(self, all_results):
        avg_result = []
        for i in range(len(all_results[0])):
            avg_result.append(np.average([result[i] for result in all_results], 0))
        for line_result in avg_result:
            self.print_list(line_result)
    
        doc = open('/home/yangruiling/mmskeleton/static/draw_data/longlife.txt', 'w')
        mean = avg_result[-1].mean()
        avg_result.append(round(mean,4))
        doc.write(str(avg_result))
        doc.close()
        return avg_result

    def print_avg_cand(self, sample_list):
        cand_lengths = []
        for sample in sample_list:
            cand_lengths.append(len(sample[1]))
        print('avg cand size:', np.average(cand_lengths))

    def print_list(self, result):
        for num in result:
            sys.stdout.write('%.3f, ' %num)
        print('')

    def output(self):
        avg_result_all_test = np.average(self.result_all_test_data, 0)      #计算加权平均数，如果有weight的话
        for result_whole_test in self.result_all_test_data:
            self.print_list(result_whole_test)
        print ("-------------------------------------------")
        self.print_list(avg_result_all_test)          #使用整个测试集进行测试的平均得分
        print ("===========================================")
        all_results = self.print_avg_results(self.all_results)

        print ('Top 1: {:.2f}%'.format(all_results[-1].mean()))

