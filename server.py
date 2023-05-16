# coding:utf-8
# 服务模块，定义web的交互逻辑
import web
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import json

# init the web server
render = web.template.render('templates/')     #指定html模版的文件路径 render对象从模版根目录查找模版文件
urls = ('/.*', 'process')      #url映射

class process:        #与url映射同名，url处理的类
    def GET(self):      #处理get请求
        return render.index()        #调用html模版，index()是模版的名字     渲染html模版文件：此处返回的html代码是网页所展示的内容
        
    def POST(self):
        x = web.input(video_file={})
        if 'video_file' in x:    #若文件存在
            # save image  保存用户上传的图片
            filedir = '/home/yangruiling/mmskeleton/resource/data_example_web'
            filepath=x.video_file.filename.replace('\\','/')
            filename=filepath.split('/')[-1]
            fout = open(filedir +'/'+ filename,'wb')
            fout.write(x.video_file.file.read())
            fout.close()

            #为视频制作导航文件(json)
            catrgory = {"categories": [filename],"annotations": {filename: {"category_id": 0}}}
            catrgory_json = json.dumps(catrgory)
            f2 = open('/home/yangruiling/mmskeleton/resource/category_annotation_example_web.json', 'w')
            f2.write(catrgory_json)
            f2.close()

            #调用ST-GCN进行动作识别，并输出动作类别
            #os.system("python mmskl.py /home/yangruiling/mmskeleton/configs/utils/build_dataset_example_web.yaml")
            os.system("python mmskl.py /home/yangruiling/mmskeleton/configs/recognition/st_gcn/dataset_example/test_web.yaml")





        # pred image   调用推理模块进行推理
        #调用recognition-test
        print("ST_GCN:")
        #os.system("python mmskl.py /home/yangruiling/mmskeleton/configs/recognition/st_gcn_aaai18/ntu-rgbd-xsub/test.yaml")
        print("lifelong:")
        #调用longlife-test
        #os.system("python /home/yangruiling/mmskeleton/mmskeleton/fewrel/test_lifelong_model.py")
        
        #获取输出数据
        f = open('/home/yangruiling/mmskeleton/static/draw_data/ST_GCN1.txt')
        st_gcn_data = f.read()
        f.close()

        st_gcn_data = ast.literal_eval(st_gcn_data)   #str转list
        st_gcn_data_shape = st_gcn_data[0][0]     #(16487, 60)
        st_gcn_label_shape = st_gcn_data[1][0]      #(16487,)
        st_gcn_Top1 = st_gcn_data[2][0]    #0.8157
        st_gcn_Top5 = st_gcn_data[3][0]    #0.9685 
        
        f = open('/home/yangruiling/mmskeleton/static/draw_data/longlife.txt')
        longlife_data = f.read()
        f.close()
        
        longlife_data = ast.literal_eval(longlife_data)
        longlife_results = longlife_data[0]   #各个分类准确率列表
        longlife_mean = longlife_data[1][0]   #平均准确率
        

        # 绘制结果图像,水平条形图方法barh()  
        # 参数一：y轴   参数二：x轴
        filedir = './static/img'     #图片保存的目录
        result_img = filedir + '/' + 'result.jpg'
        
        #解决plt中文显示的问题
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(5,3))
        price = [st_gcn_Top1,st_gcn_Top5,longlife_mean]
        xlabel = ['S_Top1', 'S_Top5','l_Top1']
        plt.bar(xlabel , price)
        plt.ylim(0,1.1)   #x轴范围
        plt.ylabel("accuracy rate")
        plt.title("ST_GCN and longlife")
        for x, y in enumerate(price):    #设置数据标签
            plt.text(x + 0.2, y - 0.1, '%s' % y)
        plt.savefig(result_img)   #存储图片

        plt.figure(figsize=(20,10))
        longlife_img = filedir + '/' + 'longlife_result.jpg'
        action_class = []
        for i in range(60):
            action_class.append(str(i+1))
        action_class = tuple(action_class)
        plt.bar(action_class, longlife_results)
        plt.ylim(0,1.1)
        plt.xlabel("class")
        plt.ylabel("accuracy rate")
        plt.title('mean = ' + str(longlife_mean))
        plt.savefig(longlife_img)   #存储图片

        return render.detect(result_img, longlife_img, 'result:')  #返回保存的结果图片

        
if __name__ == "__main__":       #启用web应用

    # start the web server
    app = web.application(urls, globals())
    app.run()     #开始创建web页面


