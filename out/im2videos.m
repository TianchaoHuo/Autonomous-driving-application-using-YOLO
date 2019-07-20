%��ͼƬת��Ϊ��Ƶ
clear all;
clc;
srcDic = uigetdir('C:\Users\Admin\Desktop\Deep Learning\deeplearning.ai-master\deeplearning.ai-master\02-�κ���ҵ\04-���Ŀ� ���������\���Ŀε����ܱ����ҵ\Car detection for Autonomous Driving\out');
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