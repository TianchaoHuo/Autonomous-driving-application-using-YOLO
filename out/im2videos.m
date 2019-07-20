%将图片转化为视频
clear all;
clc;
srcDic = uigetdir('C:\Users\Admin\Desktop\Deep Learning\deeplearning.ai-master\deeplearning.ai-master\02-课后作业\04-第四课 卷积神经网络\第四课第三周编程作业\Car detection for Autonomous Driving\out');
cd(srcDic);
allnames = struct2cell(dir('*.jpg'));
[k,len]=size(allnames);
aviobj = VideoWriter('example.avi');
aviobj.FrameRate = 2;
open(aviobj)
for i = 1:len
    name = allnames{1,i};
    frame = imread(name);
    writeVideo(aviobj,frame);
end
close(aviobj)