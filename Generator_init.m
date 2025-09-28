function [Generatorinit] = Generator_init(opdata,Syn_Machidx_g)


%%%  Bus No. xd   xq    xd'  xq'  xd''  xq''   Td0'   Tq0'  Td0''  Tq0''   H   D     SgenB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gendata=...
[
    1   1.8/9   1.7/9  0.3/9  0.55/9   0.25/9  0.25/9   8.0  0.4  0.03  0.05  6.5*9*2       0       100;
    2   1.8/9   1.7/9  0.3/9  0.55/9   0.25/9  0.25/9   8.0  0.4  0.03  0.05  6.5*9*2       0       100;
    3   1.8/9   1.7/9  0.3/9  0.55/9   0.25/9  0.25/9   8.0  0.4  0.03  0.05  6.175*9*2     0       100;
%     4   1.8/9   1.7/9  0.3/9  0.55/9   0.25/9  0.25/9   8.0  0.4  0.03  0.05  6.175*9*2     0       100;%6
];

Xd=gendata(:,2);
Xq=gendata(:,3);%
Xd1=gendata(:,4);
Xq1=gendata(:,5);
Xd11=gendata(:,6);
Xq11=gendata(:,7);
Td0=gendata(:,8);
Tq0=gendata(:,9);
Td10=gendata(:,10);
Tq10=gendata(:,11);
Tj=gendata(:,12);
D=gendata(:,13);
SgenB=gendata(:,14);

Generatorinit.Xd=Xd;
Generatorinit.Xq=Xq;
Generatorinit.Xd1=Xd1;
Generatorinit.Xq1=Xq1;
Generatorinit.Xd11=Xd11;
Generatorinit.Xq11=Xq11;
Generatorinit.Td0=Td0;
Generatorinit.Tq0=Tq0;
Generatorinit.Td10=Td10;
Generatorinit.Tq10=Tq10;
Generatorinit.Tj=Tj;
Generatorinit.D=D;
Generatorinit.SgenB=SgenB;
%% 
%  Enable  R      Ts       Tc    T3     T4     T5     dPmax     dPmin   
Govnerdata=...
[
    1     1/225   0.1     0.5    0     1.25    5.0      7.0      -7.0
    1     1/225   0.1     0.5    0     1.25    5.0      7.0      -7.0
    1     1/225   0.1     0.5    0     1.25    5.0      7.0      -7.0
%     1     1/225   0.1     0.5    0     1.25    5.0      7.0      -7.0
];

Generatorinit.Enablegov  =Govnerdata(:,1);
Generatorinit.R_G        =Govnerdata(:,2).*SgenB./100;
Generatorinit.Ts_G       =Govnerdata(:,3);
Generatorinit.Tc_G       =Govnerdata(:,4);
Generatorinit.T3_G       =Govnerdata(:,5);
Generatorinit.T4_G       =Govnerdata(:,6);
Generatorinit.T5_G       =Govnerdata(:,7);
Generatorinit.dPmax      =Govnerdata(:,8);
Generatorinit.dPmin      =Govnerdata(:,9);

%% 

Excitdata=...
[  
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
        200      10       1   9.0 -9.0
%         200      10       1   9.0 -9.0

];


Ka=Excitdata(:,1);
Tb=Excitdata(:,2);
Tc=Excitdata(:,3);
Generatorinit.Ka=Ka;
Generatorinit.Tb=Tb;
Generatorinit.Tc=Tc;
Generatorinit.Efmax=Excitdata(:,4);
Generatorinit.Efmin=Excitdata(:,5);

%% PSS
%%%%%%%%%%%%%%%%%%%
pssdata=...
[
    1   0.24    20   0.05       0.02    3       5.4     0.1    -0.1;
    1   0.24    20   0.05       0.02    3       5.4     0.1    -0.1;
    1   39      20   0.3164     0.2355  0.3164  0.2355  0.1    -0.1;
%     1   0.24    20   0.05       0.02    3       5.4     0.1    -0.1;
];
Generatorinit.EnablePSS=pssdata(:,1);
Generatorinit.kpss=pssdata(:,2);
Generatorinit.Tw_P=pssdata(:,3);
Generatorinit.T1_P=pssdata(:,4);
Generatorinit.T2_P=pssdata(:,5);
Generatorinit.T3_P=pssdata(:,6);
Generatorinit.T4_P=pssdata(:,7);
Generatorinit.Upmax=pssdata(:,8);
Generatorinit.Upmin=pssdata(:,9);
%% 

Pg   =opdata{cellstr(string(Syn_Machidx_g)),2}./SgenB;
Qg   =opdata{cellstr(string(Syn_Machidx_g)),3}./SgenB;
Vg   =opdata{cellstr(string(Syn_Machidx_g)),4};
Theta=opdata{cellstr(string(Syn_Machidx_g)),5}*pi/180;

Ug=Vg.*exp(1j*Theta);
Sg=Pg+1j*Qg;
Ig=conj(Sg./Ug);

EQ=Ug+1j*Xq.*Ig;
Delta=angle(EQ);
DEL=Delta*180/pi;
Idq=Ig.*(sin(Delta)+1j*cos(Delta));
Udq=Ug.*(sin(Delta)+1j*cos(Delta));

Ef=imag(Udq)+Xd.*real(Idq);
Eq1=imag(Udq)+Xd1.*real(Idq);
Ed1=real(Udq)-Xq1.*imag(Idq);
Eq11=imag(Udq)+Xd11.*real(Idq);
Ed11=real(Udq)-Xq11.*imag(Idq);

Vref=abs(Vg)+Ef./Ka;
Vf1=Ef.*Tb;

%
Generatorinit.Pm0=Pg;
Generatorinit.Delta=Delta;
%
Generatorinit.Eq1=Eq1;
Generatorinit.Eq11=Eq11;
Generatorinit.Eq1=Eq1;
Generatorinit.Eq11=Eq11;
Generatorinit.Ed1=Ed1;
Generatorinit.Ed11=Ed11;
%
Generatorinit.Vref=Vref;
Generatorinit.Vf1=Vf1;



end
