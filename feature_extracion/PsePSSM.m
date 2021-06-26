clear all
clc
fid=fopen('GPCR.txt');
string=fscanf(fid,'%s'); 
firstmatches=findstr(string,'>')+7;
endmatches=findstr(string,'>')-1;
firstnum=length(firstmatches);
endnum=length(endmatches);
  for k=1:firstnum-1
    j=1;
    jj=1;
    lensec(k)=endmatches(k+1)-firstmatches(k)+1;
  end
  save xuliechangdu.mat lensec
lamdashu=10;
WEISHU=662;
load('xuliechangdu.mat')
for i=1:WEISHU
    nnn=num2str(i);
    name=strcat(nnn,'.pssm');
    fid{i}=importdata(name);
end
c=cell(WEISHU,1);
for t=1:WEISHU
    clear shu d
shu=fid{t}.data;
[M,N]=size(shu);
shuju=shu(1:lensec(1,t),1:20);
d=[];
for i=1:lensec(1,t)
   for j=1:20
       d(i,j)=1/(1+exp(-shuju(i,j)));
   end
end
c{t}=d(:,:);
end
for i=1:WEISHU
[MM,NN]=size(c{i});
 for  j=1:20
   x(i,j)=sum(c{i}(:,j))/MM;
 end
end
xx=[];
sheta=[];
shetaxin=[];
for lamda=1:lamdashu;
for t=1:WEISHU
  [MM,NN]=size(c{t});
  clear xx
   for  j=1:20
      for i=1:MM-lamda
       xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
      end
      sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
   end
end
shetaxin=[shetaxin,sheta];
end
psepssm=[x,shetaxin];
xlswrite('GPCR_PsePSSM_10.xlsx',psepssm)
      