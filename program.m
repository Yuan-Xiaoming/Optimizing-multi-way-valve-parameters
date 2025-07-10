%%总主函数，基于BP神经网络与遗传算法的挖掘机多路阀动臂联流道参数寻优

%%主函数BP，导入挖掘机多路阀动臂联流道压力损失仿真数据及结构参数，通过BP神经网络来构建多路阀动臂联结构参数与目标参数之间的映射关系

clc
clear
close all
% tic


% load input.txt;
% load output.txt;
% save p.mat;
% save t.mat;
% save data2 input output;% 或者用下面那一句保存数据
% save('data3.mat', 'input','output');


%------------------------------------数据前处理---------------------------------------
% 导入数据
load data input output

% 随机排序
k=rand(1,50);% rand函数，生成1组50个随机数
[m,n]=sort(k);% sort函数，从小到大排序，m为其中一个随机数 n为它对应的序号


% 训练集与测试集
input_train=input(n(1:50),:)';   % 输入数据中1-40行的所有位作为训练集的输入
output_train=output(n(1:50),:)'; % 输出数据中1-40行的所有位作为期望输出
input_test=input(n(41:end),:)';  % 输入数据中41-end行的所有位作为测试集输入
output_test=output(n(41:end),:)';% 输出数据中41-end行的所有位期望输出
 
% 训练集数据归一化
[input_n,input_ps]=mapminmax(input_train);
[output_n,output_ps]=mapminmax(output_train);
% 【mapminmax函数】：[y,ps] =mapminmax(x,ymin,ymax)，
% x为需要归一化的输入数据，ymin、ymax为归化到的范围，不填默认归化到[-1,1]
% 返回归化后的值y和参数ps，ps在结果反归一化中，需要调用


%---------------------------BP网络创建、训练及仿真测试----------------------------
% 创建BP网络
net=newff(input_n,output_n,6,{'tansig','purelin'},'trainlm');
%【newff函数】：输入，输出，隐含层节点数，{'隐含层传输函数TF1'，'输出层传输函数TF2'}，逆向传播训练函数BTF，trainlm为BP算法训练函数
% 常用传输函数：logsig、tansig、purelin


% 设置训练参数
net.trainParam.epochs=1000;% 训练1000次
net.trainParam.lr=0.001;% 学习率
net.trainParam.goal=1e-3;% 训练残差,1%
% net.trainParam.mc=0.9;% 动量因子，默认为0.9
% net.trainParam.show=25;% 显示的间隔次数

% 开始训练网络
net=train(net,input_n,output_n);

% 保存网络
% save data_out1 net input_ps output_ps;% 保存结果，用于后续的遗传算法
w1=net.iw{1,1};% 输入层到隐含层的权值
theta1=net.b{1};% 隐含层的阈值
w2=net.lw{2,1};% 隐含层到输出层的权值
theta2=net.b{2};% 输出层的阈值



%-------------------------------------BP网络测试----------------------------------------
% 测试集数据归一化
input_nn=mapminmax('apply',input_test,input_ps);
 % apply：调用一个对象的一个方法，用另一个对象替换当前对象
% 网络预测，测试集输出
output_sim=sim(net,input_nn);
% sim为BP神经网络预测函数，函数形式：y=sim(net,x)，y为网络预测数据，x为输入数据，net为训练好的网络
% 数据反归一化得到预测数据
OUTPUT_sim=mapminmax('reverse',output_sim,output_ps);
% reverse函数用于反转在[first,last)范围内的顺序


%-------------------------------------性能评价-------------------------------------------
% 相对误差error
error = abs(OUTPUT_sim - output_test)./output_test;
% 相abs函数用于返回数字的绝对值，正数和0返回数字本身，负数返回数字的相反数。
% 决定系数R^2，回归平方和/总平方和
% （N*预测值目标值乘积之和-预测值和*目标值和）^2   /   N*预测值平方之和-预测值和的平方×N*目标值平方之和-目标值和的平方
N = size(input_test,2);
R2 = (N * sum(OUTPUT_sim .* output_test) - sum(OUTPUT_sim) * sum(output_test))^2 / ((N * sum((OUTPUT_sim).^2) - (sum(OUTPUT_sim))^2) * (N * sum((output_test).^2) - (sum(output_test))^2));
% sum：求和函数

% 结果对比
result = [output_test' OUTPUT_sim' error'];
 
 
%---------------------------------------绘图-----------------------------------------------
figure
plot(1:N,output_test,'b:*',1:N,OUTPUT_sim,'r-o')
legend('真实值','预测值')
 
xlabel('预测样本')
ylabel('压力损失')
string = {'测试集压力损失预测结果对比';['R^2=' num2str(R2)]};
title(string)
 
 
% figure(1)
% plot(1:N,output_test,'b:*')
% hold on
% plot(1:N,OUTPUT_sim,'r-o')
% legend('期望输出','预测输出')
% xlabel('预测样本','fontsize',12)
% ylabel('压力损失','fontsize',12)
% title('BP网络压力损失预测结果对比','fontsize',12)
 
% figure(2)
% plot(error,'-*')
% xlabel('预测样本','fontsize',12)
% ylabel('误差','fontsize',12)
% title('BP网络压力损失预测结果对比','fontsize',12)
 
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%主函数Genetic，通过遗传算法来求解出对应的最佳结构参数
 
% 基于遗传算法的系统极值寻优
clc
clear
close all
 
 
%----------------------------------初始化遗传算法参数-----------------------------------
%初始化参数
popsize=40;                         %种群数量
maxgen=300;                         %迭代次数
pcross=0.4;                         %交叉概率选择，0和1之间
pmutation=0.2;                      %变异概率选择，0和1之间
 
 
lenchrom=[1 1 1 1 1 1]; 
% 【定义染色体结构，染色体长度为6位，每一位都是十进制编码=不编码】
bound=[26 30;11 16;44 46;2 6;6.8 7.8;10 14];
% 6种变量的取值范围，6组2位的参数矩阵
 
 
individuals=struct('fitness',zeros(1,popsize), 'chrom',[]);
% 将种群信息定义为一个结构体。
% 【fitness项1行30列填充0；chrom项空白 后面填充30组染色体（种群）】
avgfitness=[];                      %每一代种群的平均适应度
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体（chrom）
 
 
%-----------------------------创建初始种群 计算适应度值--------------------------------
% 初始化种群
for i=1:popsize
    % 随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);% Code编码函数，采用浮点数编码并生成一个个体，chrom项第1-30行的所有位。
    x=individuals.chrom(i,:);% 赋值给x，1行6位的染色体
 
    % 计算适应度
    individuals.fitness(i)=Fit(x);% Fit适应度函数，直接调用BP网络计算函数值作为适应度值
end
 
 
% 找最好的染色体
[bestfitness bestindex]=max(individuals.fitness);% fitness项中的最大值、最大值的序号
bestchrom=individuals.chrom(bestindex,:);% chrom项bestindex所在行的染色体作为最好的染色体
 
 
avgfitness=sum(individuals.fitness)/popsize; % fitness项求均值，平均适应度
trace=[avgfitness bestfitness]; % 记录每一代进化中平均适应度和最好的适应度
 
 
%-----------------------------------迭代寻优--------------------------------------
% 进化开始
for i=1:maxgen
%     i
    % 选择
    individuals=Select(individuals,popsize);% Select选择函数
    avgfitness=sum(individuals.fitness)/popsize;% fitness项求均值，平均适应度
    %交叉
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,popsize,bound);% Cross交叉函数
    % 变异
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,popsize,[i maxgen],bound);% Mutation变异函数
    
    % 计算适应度 
    for j=1:popsize
        x=individuals.chrom(j,:);% chrom项第1-30行的所有位，赋值给x，1行6位的染色体
        individuals.fitness(j)=Fit(x); % Fit适应度函数
    end
    
  %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=max(individuals.fitness);% fitness项中的最大值、最大值的序号
    [worestfitness,worestindex]=min(individuals.fitness);% fitness项中的最小值、最小值的序号
    
    % 代替上一次进化中最好的染色体
    if newbestfitness>bestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);% chrom项newbestindex所在行的染色体作为最好的染色体
    end
    individuals.chrom(worestindex,:)=bestchrom;% 用新一代中最好的代替最差的
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/popsize;% fitness项求均值，平均适应度
    
    trace=[trace;avgfitness bestfitness]; %记录每一代进化中最好的适应度和平均适应度
end
%进化结束
 
 
%-------------------------------------结果分析----------------------------------------
[r c]=size(trace);
plot([1:r]',trace(:,2),'r-');
title('适应度曲线','fontsize',12);
xlabel('进化代数','fontsize',12);ylabel('适应度','fontsize',12);
%  axis([0,300,0.0390,0.0408])
%  set(gca,'ytick',0.0390:0.0005:0.0405)
 
% 命令行窗口显示
disp('适应度                   变量');
x=bestchrom;
disp([bestfitness x]);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%子函数Code，将变量编码成染色体
function ret=Code(lenchrom,bound)
%本函数将变量编码成染色体，用于随机初始化一个种群
% lenchrom   input : 染色体长度
% bound      input : 变量的取值范围
% ret        output: 染色体的编码值
 
 
flag=0;
while flag==0
    pick=rand(1,length(lenchrom));% rand函数，生成1组6个0-1之间的随机数
    ret=bound(:,1)'+(bound(:,2)-bound(:,1))'.*pick;% 采用浮点数编码，并生成一个个体
    % bound(:,1)bound中所有行的第1列，
    % 线性插值（最小值＋区间长度×系数），编码结果存入ret中【得到一个1行6位的染色体】
 
    flag=test(lenchrom,bound,ret);     % test函数，检验染色体的可行性
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%子函数test，解码
function flag=test(lenchrom,bound,code)
% lenchrom   input : 染色体长度
% bound      input : 变量的取值范围
% code       output: 染色体的编码值
 
x=code; %先解码
flag=1;
if (x(1)<bound(1,1))&&(x(2)<bound(2,1))&&(x(1)>bound(1,2))&&(x(2)>bound(2,2))
    flag=0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%子函数Fit，适应度函数构建
function fitness = Fit(x)
% 函数功能：计算该个体对应适应度值(BP网络的预测值)
% x           input     个体
% fitness     output    个体适应度值
 
 
load data_out net input_ps output_ps;% 载入训练好的BP网络
 
x=x';% 1行6位的染色体转置成列
 
%数据归一化
input_genetic=mapminmax('apply',x,input_ps);
 
%网络预测输出
output_genetic=sim(net,input_genetic);
 
%网络输出反归一化
OUTPUT_genetic=mapminmax('reverse',output_genetic,output_ps);% 2行1列
 
%适应度计算
fitness=50*(1/(0.5*OUTPUT_genetic(1,1)+0.5*OUTPUT_genetic(2,1)));
%定义适应度的权重：两个区域的流道压力损失各占50%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%子函数Select，遗传算法选择操作
function ret=select(individuals,sizepop)
% 本函数对每一代种群中的染色体进行选择，以进行后面的交叉和变异
% individuals input  : 种群信息
% sizepop     input  : 种群规模
% ret         output : 经过选择后的种群
 
fitness1=individuals.fitness;
sumfitness=sum(fitness1); % 适应度总和
sumf=fitness1./sumfitness; % 适应度占比（被选择的概率）
index=[]; 
for i=1:sizepop   %转sizepop次轮盘
    pick=rand;% 产生一个随机值，
    while pick==0    
        pick=rand;        
    end
    
    for i=1:sizepop    % 依次测试所有个体看
        pick=pick-sumf(i);        
        if pick<0     % 若个体的适应度占比＞随机值，则被选择【并没有使用累计概率】   
            index=[index i];
            break;  %寻找落入的区间，此次转轮盘选中了染色体i，注意：在转sizepop次轮盘的过程中，有可能会重复选择某些染色体
        end
    end
    
end
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%子函数Cross，遗传算法交叉操作
function ret=Cross(pcross,lenchrom,chrom,sizepop,bound)
%本函数完成交叉操作
% pcorss                input  : 交叉概率
% lenchrom              input  : 染色体的长度
% chrom     input  : 染色体群
% sizepop               input  : 种群规模
% ret                   output : 交叉后的染色体
 for i=1:sizepop  %每一轮for循环中，可能会进行一次交叉操作，染色体是随机选择的，交叉位置也是随机选择的，
     %但该轮for循环中是否进行交叉操作则由交叉概率决定（continue控制）
     
     pick=rand(1,2); % 随机选择两个染色体进行配对
     while prod(pick)==0
         pick=rand(1,2);
     end
     index=ceil(pick.*sizepop);% ceil函数，输出为 ≥输入值的最小整数
     
     % 交叉概率决定是否进行交叉
     pick=rand;
     while pick==0
         pick=rand;
     end
     if pick>pcross
         continue;
     end
     
     flag=0;
     while flag==0
         % 随机选择交叉位
         pick=rand;
         while pick==0
             pick=rand;
         end
         pos=ceil(pick.*sum(lenchrom)); %随机选择交叉位置，即选择第几个变量进行交叉，注意：两个染色体交叉的位置相同
         
         pick=rand; %交叉开始
         v1=chrom(index(1),pos);% chrom项30组染色体，第index1个的染色体的pos位
         v2=chrom(index(2),pos);
         chrom(index(1),pos)=pick*v2+(1-pick)*v1;% v1和v2两个基因值以一定比例混合
         chrom(index(2),pos)=pick*v1+(1-pick)*v2; %交叉结束
         
         flag1=test(lenchrom,bound,chrom(index(1),:));  %检验染色体1的可行性
         flag2=test(lenchrom,bound,chrom(index(2),:));  %检验染色体2的可行性
         if   flag1*flag2==0
             flag=0;
         else flag=1;
         end    %如果两个染色体不是都可行，则重新交叉
     end
 end
ret=chrom;
 
 
%%
% ①【从种群中随机取出两个染色体编码串进行配对，无放回的选择配对15次比较好】或者【将种群中的全部个体进行随机排序，2n-1与2n前后两两配对】
% ②判断每对是否交叉：产生一个0-1之间的随机数r，通过随机数与交叉概率比较，判断两个配对的染色体编码串是否进行交叉。若r>Pc，则进行交叉操作，否则重复步骤②判断下一对；【for i=1:2:sizepop，2n-1与2n配对】
% ③随机选择交叉位：对执行交叉的一对个体，根据编码位串长度m，随机产生[1,m]之间的一个整数k来确定染色体编码串上的交叉位；
% ④进行交叉：对执行交叉的一对个体，相互交换各自染色体交叉位上的基因值，完成交叉运算。
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%子函数Mutation，遗传算法变异操作
function ret=Mutation(pmutation,lenchrom,chrom,sizepop,pop,bound)
% 本函数完成变异操作
% pcorss                input  : 变异概率
% lenchrom              input  : 染色体长度
% chrom                 input  : 染色体群
% sizepop               input  : 种群规模
% opts                  input  : 变异方法的选择
% pop                   input  : 当前种群的进化代数和最大的进化代数信息
% ret                   output : 变异后的染色体
for i=1:sizepop   %每一轮for循环中，可能会进行一次变异操作，染色体是随机选择的，变异位置也是随机选择的，
    %但该轮for循环中是否进行变异操作则由变异概率决定（continue控制）
    
    pick=rand;% 随机选择一个染色体
    while pick==0
        pick=rand;
    end
    index=ceil(pick*sizepop);% ceil函数，输出为 ≥输入值的最小整数
    
    % 变异概率决定该轮循环是否进行变异
    pick=rand;% 允许随机数取0，判断是否进行变异
    if pick>pmutation
        continue;
    end
    
    flag=0;
    while flag==0
        % 变异位置
        pick=rand;
        while pick==0      
            pick=rand;
        end
        pos=ceil(pick*sum(lenchrom));  %随机选择了染色体变异的位置，即选择了第pos个变量进行变异
        
        v=chrom(i,pos); % chrom项30组染色体，第i个的染色体的pos位
        v1=v-bound(pos,1);% bound的第pos组第1位（最小取值）
        v2=bound(pos,2)-v;% bound的第pos组第2位（最大取值）
        pick=rand; %变异开始        
        if pick>0.5
            delta=v2*(1-pick^((1-pop(1)/pop(2))^2));
            chrom(i,pos)=v+delta;
        else
            delta=v1*(1-pick^((1-pop(1)/pop(2))^2));
            chrom(i,pos)=v-delta;
        end   %变异结束
        
        flag=test(lenchrom,bound,chrom(i,:));     %检验染色体的可行性
    end
end
ret=chrom;

