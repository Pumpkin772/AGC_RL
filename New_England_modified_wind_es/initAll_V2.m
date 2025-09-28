%% inital process
clear
w         =          1;
fB        =          50;                 
wB        =          2*pi*fB;
SgridB    =          100;

acom=100;
bcom=50;
lam=100;

SynMach_in_gidx     =    [1 2 3 4 5 6 7 8 9 10];%The node number of the synchronous generator connected to the large power grid. Please arrange it in the order in the gen matrix. 

EStorage_in_gidx    =    [2 6 8];

% AggWind_in_gidx     =    [3 4 7 8];
% %  Single fan capacity 
% AggSwgB           =    [2;2;2;2];%2MVA
% %  Number of aggregate fans; 
% AggNN             =    [340;50;50;50];%a DFIG model represent 50 DFIGs
% %  Rotor Speed of DFIG 
% AggWr0            =    [0.9;0.9;0.9;0.9];% 初始转速（pu）
% %  Initial power of fan 
% AggPwind_S   =    AggWr0.*...   % 单台风机输出功率
%     1.162.*AggWr0.^2;
% AggPwind_A   =    AggPwind_S.*AggNN.*AggSwgB; % 聚合后总有功
AggWind_in_gidx     =    [1 2];
AggWind_in_gidx2     =    [8 21];
Currentin_idx = [8 21 30:39];
%  Single fan capacity 
AggSwgB           =    [2;2];%2MVA
%  Number of aggregate fans; 
AggNN             =    [340;50];%a DFIG model represent 50 DFIGs
%  Rotor Speed of DFIG 
AggWr0            =    [0.9;0.9];% 初始转速（pu）
%  Initial power of fan 
AggPwind_S   =    AggWr0.*...   % 单台风机输出功率
1.162.*AggWr0.^2;
AggPwind_A   =    AggPwind_S.*AggNN.*AggSwgB; % 聚合后总有功
% mpcac =loadcase('P_ACdata');


[busdata,linedata]=PowerFlowData;

[opdata,opdata_all,YY_load]=powerflowcaculate(busdata,linedata);





%% Synchronous generator initialization 
Geninit=Generator_init(opdata,SynMach_in_gidx);
%% 储能变换器初始化
[ESinit] = ESconverter_init(opdata,EStorage_in_gidx);
%% DFIG initialization 
[Windinit,AVm0] = nWind_machine_init(opdata,AggWind_in_gidx,AggWr0,AggNN,AggSwgB);
[Windinit,AVm0] = nWind_machine_init(opdata_all,AggWind_in_gidx2,AggWr0,AggNN,AggSwgB);
%% 通讯延时初始化
DataDelay = ones(4,3);%双向延时，向两侧发送的数据延时
DataDelay = 0.0*DataDelay;
commun = ones(4,3);
commun = 0*commun;
%commun=[1,1,1,0];
save('parameterinit');
