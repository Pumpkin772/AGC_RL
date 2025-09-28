function [ACgirdinit,opdata] = ACgrid_init(resultsac,resultsdc,fbranchac)

nn         =max( resultsac.bus.Zone );
VSS0       =cell(nn,1);
busdata    =cell(nn,1);
linedata   =cell(nn,1);
fline         =cell(nn,1);
Currentin_idx =cell(nn,1);
ng         =cell(nn,1);


brcjoinbus=join(resultsac.branch,resultsac.bus,'LeftKeys','F_Bus','RightKeys','Bus_I');
fbrcjoinbus=join(fbranchac,resultsac.bus,'LeftKeys','F_Bus','RightKeys','Bus_I');
genjoinbus=join(resultsac.gen,resultsac.bus,'LeftKeys','Gen_Bus','RightKeys','Bus_I');

ConvJbusdc=join(resultsdc.convdc(:,1:5),resultsdc.busdc(:,{'Busdc_I','Busac_I'}),'LeftKeys','Conv_Bus','RightKeys','Busdc_I');
if ConvJbusdc.Busac_I~=0
    ConvJbusdcJbus=join(ConvJbusdc,resultsac.bus(:,{'Bus_I','Zone'}),'LeftKeys','Busac_I','RightKeys','Bus_I');
else
    ConvJbusdcJbus=[];
end

opdata=genjoinbus(:,{'Gen_Bus','PG','QG','Vm','Va','Gen_idx','GWType','Zone'});

opdata{:,1} = (1:height(opdata)).';



for ii=1:1:nn

    Bus_ii=resultsac.bus(resultsac.bus.Zone==ii,:);
    
    Branchac_ii=brcjoinbus(brcjoinbus.Zone==ii,[1:17,27]);
   

    
    fBranchac_ii=fbrcjoinbus(fbrcjoinbus.Zone==ii,[1:17,27]);

    
    opdataii=opdata(opdata.Zone==ii,:);
    if ~isempty(ConvJbusdcJbus)
       ConvJbusdcJbusii=ConvJbusdcJbus(ConvJbusdcJbus.Zone==ii,:);
       Currentin_idxii=str2num(char( [opdataii.Properties.RowNames;cellstr(string(ConvJbusdcJbusii.Busac_I))] ));
    else
       Currentin_idxii=str2num(char( [opdataii.Properties.RowNames] ));
    end

    
    %% 

    OrBusI=Bus_ii{:,'Bus_I'};

    Bus_ii{:,'OrBusI'}=OrBusI;
    Bus_ii{:,'Bus_I'}=[1:1:size(Bus_ii{:,1},1)].';
    Bus_ii.Properties.RowNames=cellstr(string([1:1:size(Bus_ii{:,1},1)].'));


   
    for jj=1:size(Currentin_idxii,1)
    Currentin_idxii(jj,1)= Bus_ii{ Currentin_idxii(jj,1)==Bus_ii.OrBusI, 'Bus_I' };
    end
    
    
    for jj=1:size(Branchac_ii,1)
        Branchac_ii{jj,'F_Bus'}= Bus_ii{ Branchac_ii{jj,'F_Bus'}==Bus_ii.OrBusI,'Bus_I' };
        Branchac_ii{jj,'T_Bus'}= Bus_ii{ Branchac_ii{jj,'T_Bus'}==Bus_ii.OrBusI,'Bus_I' };
    end
    
    
    for jj=1:size(fBranchac_ii,1)
        fBranchac_ii{jj,'F_Bus'}= Bus_ii{ fBranchac_ii{jj,'F_Bus'}==Bus_ii.OrBusI,'Bus_I' };
        fBranchac_ii{jj,'T_Bus'}= Bus_ii{ fBranchac_ii{jj,'T_Bus'}==Bus_ii.OrBusI,'Bus_I' };
    end
    
    
    
    %% 
%VSS0=[编号，电压幅值，相角(°)，向量（复数）]

    VSS0ii              =      Bus_ii(:,[1,8,9]);
    Phasor_V            =      VSS0ii.Vm.*exp(1i*VSS0ii.Va*pi./180);
    VSS0ii{:,'Phasor_V'}=      Phasor_V;

%  busdata=[Bus_No. Volt_Mag   Volt_angle    Pg   Qg   Pl   Ql  Gshunt Bshunt Bus type (=1 swing bus, =2 PV bus, =3 PQ bus)]
    busdataii           =      Bus_ii(:,[1,8,9,1,1,3,4,5,6,2]);
    busdataii{:,[6,7]}  =      busdataii{:,[6,7]}./resultsac.baseMVA;%
    busdataii{:,10}     =      4-busdataii{:,10};
    
%  linedata=[From Bus     To Bus    R (p.u.)  X (p.u.)  B/2 (p.u.)  tap (tap=0 for line)]
    linedataii          =      Branchac_ii(:,[1,2,3,4,5,9,11]);

    if ( ~isempty(fBranchac_ii) )
    flineii             =      fBranchac_ii(:,[1,2,3,4,5,9]);
    else
    flineii             =      fBranchac_ii(:,[1,2,3,4,5,9]);
    flineii{1,1:6}      =      [999 999 15 15 15 0];    
    end

    VSS0(ii)            =       {VSS0ii};
    busdata(ii)         =       {busdataii};
    linedata(ii)        =       {linedataii};
    fline(ii)           =       {flineii};
    Currentin_idx(ii)   =       {Currentin_idxii};
    
    ng(ii)              =       {size(Currentin_idxii,1)};
end

  
%% Identify short-circuited nodes 
FBUS_1 = fline{1}{:,1};
% FBUS_2 = fline{2}{:,1};
%FBUS   = [FBUS_1];
FBUS=[1,12];

%%
ACgirdinit.VSS0=VSS0;
ACgirdinit.busdata=busdata;
ACgirdinit.linedata=linedata;
ACgirdinit.fline=fline;
ACgirdinit.FBUS=FBUS;
ACgirdinit.Currentin_idx=Currentin_idx;
ACgirdinit.ng=ng;
end