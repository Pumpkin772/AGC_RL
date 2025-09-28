function [baseMVA, bus, gen, branch] = ACdata
%case 5 nodes    Power flow data for 5 bus, 2 generator case.
%   Please see 'help caseformat' for details on the case file format.
%
%   case file can be used together with dc case files "case5_stagg_....m"
%
%   Network data from ...
%   G.W. Stagg, A.H. El-Abiad, "Computer methods in power system analysis",
%   McGraw-Hill, 1968.
%
%   MATPOWER case file data provided by Jef Beerten.

%% MATPOWER Case Format : Version 1
%%-----  Power Flow Data  -----%%
%% system MVA base
baseMVA = 100;

%% bus data
%  1        2       3       4       5   6       7       8       9   10         11       12  13
%  BUS_I, BUS_TYPE, PD,     QD,     GS, BS,  BUS_AREA,  VM,     VA, BASE_KV, ZONE,    VMAX, VMIN
%	bus_i	type	Pd      Qd      Gs	Bs      area	Vm      Va	baseKV	 zone     Vmax	Vmin
%  母线编号 类型 有功负荷 无功负荷            区域编号 电压幅值和相角初值
bus = [
	1       3       0       0       0	0       1       1.05     0      0       1       1.1     0.9;%发电机 参考节点
	2       2       0       0   	0	0       1       1.05 	 0      0       1       1.1     0.9;%发电机 PV节点
	3       2       0   	0       0	0       1       1.05 	 0   	0       1       1.1     0.9;%发电机 PV节点
	4       1       0   	0-188   	0	0       1       1.05	 0      0       1       1.1     0.9;%风机 PQ节点
	5       1       0       0       0	0       1       1.00	 0    	0       1       1.1     0.9;
	6       1       0       0    	0	0       1       1.00	 0      0       1       1.1     0.9;
	7       1       967     -100    0	0       1       1.00	 0      0       1       1.1 	0.9;
	8       1       0       0       0	0       1       1.00	 0      0       1       1.1 	0.9;   
    9       1       1767  	-250 	0	0       1       1.00	 0      0       1       1.1     0.9;
	10      1       0       0       0	0       1       1.00	 0      0       1       1.1 	0.9;
	11      1       0       0       0	0       1       1.00	 0      0       1       1.1     0.9;	
  
    % 增加的储能节点
    12      1       0       0       0   0       1       1.05	 0      0       1       1.1     0.9;
    13      1       0       0       0   0       1       1.05	 0      0       1       1.1     0.9;
    14      1       0       0       0   0       1       1.05	 0      0       1       1.1     0.9;
    15      1       0       0       0   0       1       1.05	 0      0       1       1.1     0.9;

% 增加风机节点
    16      1       0       0       0   0       1       1.00	 0      0       1       1.1     0.9;   
    
    ];
%-400MVar；250MW；120MW，140Mvar；
%% generator data
%   1       2       3       4      5    6       7           8           9   10
% GEN_BUS, PG,    QG,      QMAX, QMIN, VG,      MBASE,    GEN_STATUS, PMAX, PMIN,
%	bus	  Pg      Qg	   Qmax	 Qmin	Vg      mBase       status	Pmax	Pmin
% bus：发电机连接的母线编号 Pg：发电机初始有功出力（潮流输入） Qg：发电机初始无功出力 Vg：发电机母线的电压目标值（用于 PV 节点）
gen = [
    1    600     0   1000     -400    1.03     100     1       1000     0;
    2    700     0    5000     -400    1.01     100     1       1000 	 0;
    3    800     0    4000     -400    1.03     100 	1       1000     0;
    4    650     0    2400     -400    1.01     100 	1       1000     0;
    
    12     0      0     1000    -1000   1.01    100     1       1000    -1000;
    13     0      0     1000    -1000   1.01    100     1       1000    -1000;
    14     0      0     1000    -1000   1.01    100     1       1000    -1000;    
    15     0      0     1000    -1000   1.01    100     1       1000    -1000;
    
    16     0      0     1000    -1000   1.01    100     1       1000    0;
    ];

%% branch data
%  1        2       3       4           5           6       7       8       9       10      11          12       13
%  F_BUS,  T_BUS, BR_R,    BR_X,      BR_B,        RATE_A, RATE_B,  RATE_C, TAP,   SHIFT,  BR_STATUS, ANGMIN, ANGMAX,
%	fbus	tbus	r       x           b           rateA	rateB	rateC	ratio	angle	status
%  起止母线编号                                                           变比与移相角（用于变压器）
branch = [
    1       5       0       0.01670     0           0       0         0      1      0      1;
    2       6       0       0.01670     0           0       0         0      1   	0       1;
    3       11      0       0.01670     0           0       0         0      1   	0       1;
    4       10      0       0.01670     0           0       0         0      1   	0       1;
    5       6       0.0025	0.025       0.04375     0       0         0      0   	0       1;
    6       7       0.001	0.01        0.0175      0       0         0      0   	0       1;
    7       8       0.011   0.11        0.1925      0       0         0      0      0      1;
    7       8       0.011   0.11        0.1925      0       0         0      0      0      1;
    8       9       0.011   0.11        0.1925      0       0         0      0      0      1;
    8       9       0.011   0.11        0.1925      0       0         0      0      0      1;
    9       10      0.001	0.01        0.0175      0       0         0      0   	0       1;
    10      11      0.0025	0.025       0.04375     0       0         0      0   	0       1;
    
    12      5       0       0.01670     0           0       0         0      1      0      1;
    13      8       0       0.01670     0           0       0         0      1   	0       1;
    14      10      0       0.01670     0           0       0         0      1   	0       1;   
    15      9       0       0.01670     0           0       0         0      1   	0       1;
    
    16      9       0       0.01670     0           0       0         0      1   	0       1; 
    ];
return;
