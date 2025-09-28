function [TAmpc] = mpcTVSA(mpc,Usingchar)
% 请输入'Mac_A2T','Mdc_A2T','Rac_A2T','Rdc_A2T','DC_T2A','AC_T2A'
switch Usingchar
    case 'Mac_A2T'
        TAmpc=mpcacnum2table(mpc);
    case 'Mdc_A2T'
        TAmpc=mpcdcnum2table(mpc);
    case 'Rac_A2T'
        TAmpc=resultacnum2table(mpc);
    case 'Rdc_A2T'
        TAmpc=resultdcnum2table(mpc);
    case 'DC_T2A'
        TAmpc=tabledc2dc(mpc);
    case 'AC_T2A'
        TAmpc=tableac2ac(mpc);
    otherwise
        error('Undefined')
        
end



end



%% 对ac的转换
function [tablempcac] = mpcacnum2table(mpc_ac)
tablempcac.version=mpc_ac.version;
tablempcac.baseMVA=mpc_ac.baseMVA;
Table_bus=array2table(mpc_ac.bus,...
    'VariableNames',{'Bus_I','Bus_type','PD','QD','GS','BS','Area','Vm','Va','BasekV','Zone','Vmax','Vmin'});
BusI= cellstr(string(mpc_ac.bus(:,1)));
Table_bus.Properties.RowNames=BusI;
tablempcac.bus=Table_bus;

Table_gen=array2table(mpc_ac.gen,...
    'VariableNames',{'Gen_Bus','PG','QG','Qmax','Qmin','VG','mbase','Gen_status','Pmax','Pmin',...
    'Pc1','Pc2','Qc1max','Qc1min','Qc2max','Qc2min','RampACG','Ramp10','Ramp30','RampQ','APF'});

Gen_Bus= cellstr(string(mpc_ac.gen(:,1)));
Table_gen.Properties.RowNames=Gen_Bus;
tablempcac.gen=Table_gen;

Table_branch=array2table(mpc_ac.branch,...
    'VariableNames',{'F_Bus','T_Bus','R','X','B','RateA','RateB','RateC',...
    'Tap','Shift','Brc_status','Angmin','Angmax'});
Brch_num=cellstr(string([1:1:size(mpc_ac.branch,1)].'));
Table_branch.Properties.RowNames=Brch_num;
tablempcac.branch=Table_branch;

end

%% 对dc的转换
function [tablempcdc] = mpcdcnum2table(mpc_dc)

tablempcdc.version=mpc_dc.version;
tablempcdc.baseMVAac=mpc_dc.baseMVAac;
tablempcdc.baseMVAdc=mpc_dc.baseMVAdc;
tablempcdc.pol=mpc_dc.pol;

Table_DCbus=array2table(mpc_dc.busdc,...
    'VariableNames',{'Busdc_I','Busac_I','GridDC','Pdc','Vdc','Base_kVDC','VdcMAX','VdcMIN','Cdc'});
BusdcI= cellstr(string(mpc_dc.busdc(:,1)));
Table_DCbus.Properties.RowNames=BusdcI;
tablempcdc.busdc=Table_DCbus;

Table_Conv=array2table(mpc_dc.convdc,...
    'VariableNames',{'Conv_Bus','ConvType_DC','ConvType_AC','Pconv','Qconv','Vconv',...
    'Rtf','Xtf','Bf','Rconv','Xconv','BaseKVC','VcMAX','VcMIN','IcMAX','ConvStatus',...
    'LossA','LossB','LossRec','LossInv'...
    'Droop','Pdcset','Vdcset','DVdcset'});
ConvdcI= cellstr(string(mpc_dc.convdc(:,1)));
Table_Conv.Properties.RowNames=ConvdcI;
tablempcdc.convdc=Table_Conv;

Table_branchdc=array2table(mpc_dc.branchdc,...
    'VariableNames',{'F_Bus','T_Bus','R','L','C','RateA','RateB','RateC',...
    'BrcDC_status'});
Brchdc_num=cellstr(string([1:1:size(mpc_dc.branchdc,1)].'));
Table_branchdc.Properties.RowNames=Brchdc_num;
tablempcdc.branchdc=Table_branchdc;
end

%% 对ac的结果转换
function [tablempcac] = resultacnum2table(mpc_ac)
tablempcac=mpc_ac;
Table_bus=array2table(mpc_ac.bus,...
    'VariableNames',{'Bus_I','Bus_type','PD','QD','GS','BS','Area','Vm','Va','BasekV','Zone','Vmax','Vmin'});
BusI= cellstr(string(mpc_ac.bus(:,1)));
Table_bus.Properties.RowNames=BusI;
tablempcac.bus=Table_bus;


Table_gen=array2table(mpc_ac.gen,...
    'VariableNames',{'Gen_Bus','PG','QG','Qmax','Qmin','VG','mbase','Gen_status','Pmax','Pmin',...
    'Pc1','Pc2','Qc1max','Qc1min','Qc2max','Qc2min','RampACG','Ramp10','Ramp30','RampQ','APF'});

Gen_Bus= cellstr(string(mpc_ac.gen(:,1)));
Table_gen.Properties.RowNames=Gen_Bus;
tablempcac.gen=Table_gen;

Table_branch=array2table(mpc_ac.branch,...
    'VariableNames',{'F_Bus','T_Bus','R','X','B','RateA','RateB','RateC',...
    'Tap','Shift','Brc_status','Angmin','Angmax','PF','QF','PT','QT'});
Brch_num=cellstr(string([1:1:size(mpc_ac.branch,1)].'));
Table_branch.Properties.RowNames=Brch_num;
tablempcac.branch=Table_branch;



end

%% 对dc结果进行转换
function [tablempcdc] = resultdcnum2table(mpc_dc)

tablempcdc=mpc_dc;
Table_DCbus=array2table(mpc_dc.busdc,...
    'VariableNames',{'Busdc_I','Busac_I','GridDC','Pdc','Vdc','Base_kVDC','VdcMAX','VdcMIN','Cdc'});
BusdcI= cellstr(string(mpc_dc.busdc(:,1)));
Table_DCbus.Properties.RowNames=BusdcI;
tablempcdc.busdc=Table_DCbus;

Table_Conv=array2table(mpc_dc.convdc,...
    'VariableNames',{'Conv_Bus','ConvType_DC','ConvType_AC','Pconv','Qconv','Vconv',...
    'Rtf','Xtf','Bf','Rconv','Xconv','BaseKVC','VcMAX','VcMIN','IcMAX','ConvStatus',...
    'LossA','LossB','LossRec','LossInv'...
    'Droop','Pdcset','Vdcset','DVdcset'...
    'Vmc','Vac','Pcconv','Qcconv','Pcloss','Vmf','Vaf','Pfil','Qconvf','Qcconvf'});
ConvdcI= cellstr(string(mpc_dc.convdc(:,1)));
Table_Conv.Properties.RowNames=ConvdcI;
tablempcdc.convdc=Table_Conv;

Table_branchdc=array2table(mpc_dc.branchdc,...
    'VariableNames',{'F_Bus','T_Bus','R','L','C','RateA','RateB','RateC',...
    'BrcDC_status','PFdc','PTdc'});
Brchdc_num=cellstr(string([1:1:size(mpc_dc.branchdc,1)].'));
Table_branchdc.Properties.RowNames=Brchdc_num;
tablempcdc.branchdc=Table_branchdc;
end
%% 将AC表转成矩阵
function [mpcac] = tableac2ac(tablempcac)

mpcac=tablempcac;
mpcac.bus=tablempcac.bus{:,:};
mpcac.gen=tablempcac.gen{:,:};
mpcac.branch=tablempcac.branch{:,:};
end

%% 将DC表转成矩阵
function [mpcdc] = tabledc2dc(tablempcdc)

mpcdc=tablempcdc;
mpcdc.busdc=tablempcdc.busdc{:,:};
mpcdc.convdc=tablempcdc.convdc{:,:};
mpcdc.branchdc=tablempcdc.branchdc{:,:};
end