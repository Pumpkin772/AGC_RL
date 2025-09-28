%% Common parameters
clear;

w         =          1;
fB        =          50;                   
wB        =          2*pi*fB;



SynMach_in_gidx     =    [1 2 3];%The node number of the synchronous motor connected to the large power grid. 
                                    % Please arrange it in the order in the gen matrix.   
%  储能对应的节点
EStorage_in_gidx     =   [12 13 14 15];
AggWind_in_gidx     =    [4;16];%The node number of the wind turbine connected to the large power grid. Please arrange it in the order in the gen matrix.

%  Single fan capacity 
AggSwgB           =    [2;2];%2MVA
%  Number of aggregate fans 
AggNN             =    [340;50];%a DFIG model represent 50 DFIGs
%  Rotor Speed of DFIG 
AggWr0            =    [0.9;0.9];% 初始转速（pu）
%  Initial power of fan 
AggPwind_S   =    AggWr0.*...   % 单台风机输出功率
    1.162.*AggWr0.^2;
AggPwind_A   =    AggPwind_S.*AggNN.*AggSwgB; % 聚合后总有功




mpcdc =loadcasedc('P_DCdata');
mpcac =loadcase('P_ACdata');


[resultsac,resultsdc,converged]=powerflowcaculate(mpcac,mpcdc,AggPwind_A,AggWind_in_gidx,SynMach_in_gidx,EStorage_in_gidx);


  

if(converged~=1)
    error('潮流计算不收敛')
end
clear converged

SgridB    =     resultsac.baseMVA;   % AC网络的基准功率
SdcgridB  =     resultsdc.baseMVAdc;   %DC网络的基准功率




%% AC network initialization 

fbranchac = resultsac.branch(1,:);  % Set up short line 

[ACgirdinit,opdata]=ACgrid_init(resultsac,resultsdc,fbranchac);
 ACgirdinit.amp = 0.0;



%% Synchronous motor initialization 
Geninit=Generator_init(opdata,SynMach_in_gidx);

%% DFIG initialization 
[Windinit,AVm0] = nWind_machine_init(opdata,AggWind_in_gidx,AggWr0,AggNN,AggSwgB);

%% 储能变换器初始化
[ESinit] = ESconverter_init(resultsac,EStorage_in_gidx);

%% 通讯延时初始化
DataDelay = ones(length(EStorage_in_gidx),2);%双向延时，向两侧发送的数据延时
DataDelay = 0.0*DataDelay;



commun=[1,1,1,0];
comdisconnect = 4;
flag1=0.5881;
flag2=0.8977;
flag3=0.8915;
amp=0;
type=0.5881;
save('parameterinit');