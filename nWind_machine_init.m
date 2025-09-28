function [Windinit,Vm0] = nWind_machine_init(opdata,AggWind_in_gidx,Wr0,NN,SwgBi)
%% 

%% 
%    
%   
%    1Wind_idx    2Lm         3Ls1     4Lr1     5Rs   6Rr   7NPf   8H   9Cw   10Lg   11Rg   12Vdc   
Wind_machine_data=...
[    
        1         2.67      0.049    0.065       0    0      1     8     50    0.1   0       2.828   
        2         2.67      0.049    0.065       0    0      1     8     50    0.1   0       2.828 
];
%
%，
%        1            23       45         67           89       1011     1213        1415      1617
%    风机序号     锁相环PI   转子d内环   转子d外环   转子q内环  转子q外环  网侧d内环  网侧d外环  网侧q内环
Wind_mach_PI=...
[
        1          100,6000    0.95,1     -0.5,-1.2    0.1,0.1    0.5,5    2,5        1,20    2,5
        2          100,6000    0.95,1     -0.5,-1.2    0.1,0.1    0.5,5    2,5        1,20    2,5
];

%% 

         Windinit.Vdc_w      =        Wind_machine_data(:,12);   
         Windinit.Vdc0       =        Wind_machine_data(:,12); 
         
         
%%      

        Lm        =     Wind_machine_data(:,2)   ;             
        Ls        =     Wind_machine_data(:,3)+Lm; %0.18
        Lr        =     Wind_machine_data(:,4)+Lm;         
        Rs        =     Wind_machine_data(:,5)   ;%0.078;
        Rr        =     Wind_machine_data(:,6)   ;%0.044;
        NPf       =     Wind_machine_data(:,7)   ;
        H         =     Wind_machine_data(:,8)   ;%6.1

        
        Ta0       =     Lr./(Rr.*100.*pi);%
        Xa        =     Ls;
        Xa1       =     Ls-(Lm.^2)./Lr;


        Za        =      cell(size(Wind_machine_data,1),1);
        for  ii=1:size(Wind_machine_data,1)
            Za(ii)    =     {inv([Rs(ii) -Xa1(ii);Xa1(ii) Rs(ii)])};
        end

Windinit.Lm  =  Lm;
Windinit.Lr  =  Lr;
Windinit.Ls  =  Ls;
Windinit.Rs  =  Rs;
Windinit.Rr  =  Rr;
Windinit.NPf =  NPf;
Windinit.H   =  H;
Windinit.Ta0 =  Ta0;
Windinit.Xa  =  Xa;
Windinit.Xa1 =  Xa1;
Windinit.Za  =  Za;
Windinit.SwgBi=SwgBi;
        
%% 

Windinit.Kppll_wind    =     Wind_mach_PI(:,2);
Windinit.Kipll_wind    =     Wind_mach_PI(:,3);

%% 
        %d-axis control 
Windinit.KpP_rw        =     Wind_mach_PI(:,6);
Windinit.KiP_rw        =     Wind_mach_PI(:,7);
Windinit.KpId_rw       =     Wind_mach_PI(:,4);
Windinit.KiId_rw       =     Wind_mach_PI(:,5);
        %q-axis control 
Windinit.KpQ_rw        =     Wind_mach_PI(:,10);
Windinit.KiQ_rw        =     Wind_mach_PI(:,11);
Windinit.KpIq_rw       =     Wind_mach_PI(:,8);
Windinit.KiIq_rw       =     Wind_mach_PI(:,9);

%% 
Windinit.C_w           =      Wind_machine_data(:,9);

%% 
         Lg            =      Wind_machine_data(:,10);
         Rg            =      Wind_machine_data(:,11);

%d-axis control 
Windinit.KpDC_gw       =      Wind_mach_PI(:,14);
Windinit.KiDC_gw       =      Wind_mach_PI(:,15);

Windinit.KpId_gw       =      Wind_mach_PI(:,12);
Windinit.KiId_gw       =      Wind_mach_PI(:,13);

%q-axis control 
Windinit.KpIq_gw       =      Wind_mach_PI(:,16);
Windinit.KiIq_gw       =      Wind_mach_PI(:,17);

Windinit.Lg            =      Lg;
Windinit.Rg            =      Rg;

%% ===============================================================

%% 

         
         Pw          =    opdata{cellstr(string(AggWind_in_gidx)),2}./(NN.*SwgBi);%
         Qw          =    opdata{cellstr(string(AggWind_in_gidx)),3}./(NN.*SwgBi);
         Um          =    opdata{cellstr(string(AggWind_in_gidx)),4};
         Atheta      =    opdata{cellstr(string(AggWind_in_gidx)),5}*pi/180;
 
         
         PwA        =    opdata{cellstr(string(AggWind_in_gidx)),2}./(NN.*SwgBi);
         option1    =    optimset('display','off','TolFun',1e-1,'TolX',1e-1,'Display','off');
         option2    =    optimset('MaxFunEvals',1e8,'TolFun',1e-12,'TolX',1e-12,'Display','off');
         
         F=@(wind_speed,Pw,Wr)...
         0.95*1/720.*(25.52.*(wind_speed./(99.4945.*Wr)-0.0035)-1.1)...
         .*exp( -12.5.*(wind_speed./(99.4945.*Wr)-0.0035) ).*wind_speed.^3-Pw;
        
         Vm0=zeros( size(Wind_machine_data,1) , 1 );
         
         for ii=1:size(Wind_machine_data,1)
         Vm0(ii)        =    fsolve(F,8,option1,PwA(ii),Wr0(ii));
         Vm0(ii)        =    fsolve(F,Vm0(ii),option2,PwA(ii),Wr0(ii));
         end
          
Windinit.Atheta=Atheta;
Windinit.NN    =NN;
Windinit.Wr0   =Wr0;

%% 
        Ps0            =    Pw./(Wr0);
        Qs0            =    Qw;
        
Windinit.Qs0           =    Qs0;        
%% 

         Udq_PCC         =       Um+1i*0;
         Ud_PCC          =       real(Udq_PCC);
         Uq_PCC          =       imag(Udq_PCC);
         Isdq            =       conj(-(Ps0+1i*Qs0)./Udq_PCC);
         Isd             =       real(Isdq);%%定子电流
         Isq             =       imag(Isdq);
         Ed_ref          =       Ud_PCC-Rs.*Isd+Xa1.*Isq;
         Eq_ref          =       Uq_PCC-Rs.*Isq-Xa1.*Isd;
         
         Uxy_PCC         =       Um.*cos(Atheta)+1j.*Um.*sin(Atheta);
         Ux_PCC          =       real(Uxy_PCC);
         Uy_PCC          =       imag(Uxy_PCC);
         Isxy            =       conj(-(Ps0+1i*Qs0)./Uxy_PCC);
         Isx             =       real(Isxy);%%定子电流
         Isy             =       imag(Isxy);
         Ex_ref          =       Ux_PCC-Rs.*Isx+Xa1.*Isy;
         Ey_ref          =       Uy_PCC-Rs.*Isy-Xa1.*Isx;         
         
         Vrd             =       (Lr./Lm).*((Rr./Lr).*(Eq_ref-(Xa-Xa1).*Isd)+(1-Wr0).*Ed_ref);
         Vrq             =       (Lr./Lm).*(-(Rr./Lr).*(Ed_ref+(Xa-Xa1).*Isq)+(1-Wr0).*Eq_ref);
         
         Ird             =       (1./Lm).*Eq_ref-(Lm./Lr).*Isd;
         Ird_ref         =       Ird;
         Irq             =      -(1./Lm).*Ed_ref-(Lm./Lr).*Isq;
         Irq_ref         =       Irq;
         
         Pr              =       Vrd.*Ird+Vrq.*Irq;
         

         
Windinit.Ird_ref = Ird_ref;
Windinit.Irq_ref = Irq_ref;   
Windinit.Ex_ref = Ex_ref;
Windinit.Ey_ref = Ey_ref;
Windinit.Isx = Isx;
Windinit.Isy = Isy;

        
%% 
        
         Q_gsc           =        0;
         Icdq            =        conj( (Pr+1j*Q_gsc)./Udq_PCC);
         Icd             =        real(Icdq);
         Icq             =        imag(Icdq);

         
         Vcd_ref         =        Um-Rg.*Icd+Lg.*Icq;
         Vcq_ref         =        0 -Rg.*Icq-Lg.*Icd;
         
%          clear Pr Icd Icq Icdq Udq_PCC
          
         Uxy_PCC            =        Um.*cos(Atheta)+1j.*Um.*sin(Atheta);
         Ux_PCC             =        real(Uxy_PCC);
         Uy_PCC             =        imag(Uxy_PCC);
         Icxy               =        conj( (Pr+1j*Q_gsc)./Uxy_PCC);
         Icx0               =        real(Icxy);
         Icy0               =        imag(Icxy);
         
         
Windinit.Icd_ref = Icd; 
Windinit.Icq_ref = Icq; 
Windinit.Vcd_ref = Vcd_ref; 
Windinit.Vcq_ref = Vcq_ref; 
Windinit.Icx0 = Icx0; 
Windinit.Icy0 = Icy0; 

end

%% end
