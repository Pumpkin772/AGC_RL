function [baseMVAac, baseMVAdc, pol, busdc, convdc, branchdc]= P_DCdata

%dc case 3 nodes    dc power flow data for 3 node system
%
%   3 node system (voltage droop controlled) can be used together with 
%   ac case files 'case5_stagg.m' and 'case'3_inf.m'
%   
%   Network data based on ...
%   J. Beerten, D. Van Hertem, R. Belmans, "VSC MTDC systems with a 
%   distributed DC voltage control ? a power flow approach", in IEEE 
%   Powertech2011, Trondheim, Norway, Jun 2011.
%
%   MATACDC case file data provided by Jef Beerten.


%% system MVA base
baseMVAac = 100;
baseMVAdc = 100;

%% dc grid topology
pol=1;  % numbers of poles (1=monopolar grid, 2=bipolar grid)

%% bus data
%    1      2        3       4   5       6           7           8       9
%BUSDC_I, BUSAC_I, GRIDDC, PDC, VDC,  BASE_KVDC,   VDCMAX,   VDCMIN,     CDC
busdc = [
    1       0       1       0       1       400         1.1     0.9     3;
    2       0       1       0       1       400         1.1     0.9     3; 

];


%% converters
%     1       2          3    4         5     6          7       8      9     10        11          12          13   14      15      16        17     18    19       20         21          22      23        24
% %   busdc_i type_dc type_ac P_g       Q_g   Vtar      rtf     xtf     bf     rc     	xc       basekVac    Vmmax   Vmmin   Imax    status   LossA LossB  LossCrec LossCinv  droop      Pdcset    Vdcset  dVdcset
%CONV_BUS, CONVTYPE_DC,CONVTYPE_AC, PCONV, QCONV, VCONV, RTF, XTF, BF, RCONV, XCONV,BASEKVC, VCMAX, VCMIN, ICMAX, CONVSTATUS, LOSSA, LOSSB, LOSSCR, LOSSCI, DROOP, PDCSET, VDCSET, DVDCSET
convdc = [ 
        1       1       1       0      0      1        0.000  0.0000  0.0  0.002   0.085      220        1.1     0.9     1.1     1       0.000    0.000  0.000    0.000      0.0050    -58.6274   1.0079   0  ;
        2       2       1       0      0      1        0.000  0.0000  0.0  0.002   0.085      220        1.1     0.9     1.1     1       0.000    0.000  0.000    0.000      0.0070     21.9013   1.0000   0  ;
];

%% branches
%    1      2       3             4         5         6        7        8       9
%   fbusdc  tbusdc  r             l          c       rateA   rateB    rateC    status
%   F_BUSDC,T_BUSDC,BRDC_R,     BRDC_L,    BRDC_C, RATEDC_A, RATEDC_B,RATEDC_C, BRDC_STATUS
branchdc = [  
    1       2       0.0007125          0.0183703           0.618277/2       100     100     100     1;
 ];
