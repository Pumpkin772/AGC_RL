%% inital process
clear
w         =          1;
fB        =          50;                 
wB        =          2*pi*fB;
SgridB    =          100;


SynMach_in_gidx     =    [1 2 3 4 5 6 7 8 9 10];%The node number of the synchronous generator connected to the large power grid. Please arrange it in the order in the gen matrix. 





% mpcac =loadcase('P_ACdata');


[busdata,linedata]=PowerFlowData;

[opdata,YY_load]=powerflowcaculate(busdata,linedata);





%% Synchronous generator initialization 
Geninit=Generator_init(opdata,SynMach_in_gidx);



% R=0.0001;
% X=0.016667;
% U=1;