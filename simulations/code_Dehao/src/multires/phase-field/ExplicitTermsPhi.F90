!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         ! 
!    FILE: ExplicitTermsPhi.F90                           !
!    CONTAINS: subroutine ExplicitTermsPhi                !
!                                                         ! 
!    PURPOSE: Compute the non-linear terms associated to  !
!     the phase-field variable.                           !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine ExplicitTermsPhi
    use param
    use mgrd_arrays, only: phi,hphi,tempr,sal
    use decomp_2d, only: xstartr,xendr
    implicit none
    integer :: jc,kc,ic
    integer :: jm,jp,im,ip
    real    :: nlphi
    real    :: udzrq,udyrq
    real    :: dyyp,dzzp
    real    :: pf_B, bcl

    pf_B = pf_D / (pf_eps)**2

    udzrq=dzqr
    udyrq=dyqr

    do ic=xstartr(3),xendr(3)
        im=ic-1
        ip=ic+1
        do jc=xstartr(2),xendr(2)
            jm=jc-1
            jp=jc+1
            do kc=1,nxmr
                ! yy second derivative of phi
            if(sidewall.eq.1) then
               if(jc.eq.1) then
                  dyyp = (phi(kc,jp,ic) - 1.0*phi(kc,jc,ic) + 0.0d0        )*udyrq
               elseif(jc.eq.nymr) then
                  dyyp = (0.0d0         - 1.0*phi(kc,jc,ic) + phi(kc,jm,ic))*udyrq
               else
                  dyyp = (phi(kc,jp,ic) - 2.0*phi(kc,jc,ic) + phi(kc,jm,ic))*udyrq
               endif
            else
               dyyp = (phi(kc,jp,ic) - 2.0*phi(kc,jc,ic) + phi(kc,jm,ic))*udyrq
            endif

                ! zz second derivative of phi
            if(sidewall.eq.1) then
               if(ic.eq.1) then
                  dzzp = (phi(kc,jc,ip) - 1.0*phi(kc,jc,ic) + 0.0d0        )*udzrq
               elseif(ic.eq.nzmr) then
                  dzzp = (0.0d0         - 1.0*phi(kc,jc,ic) + phi(kc,jc,im))*udzrq
               else
                  dzzp = (phi(kc,jc,ip) - 2.0*phi(kc,jc,ic) + phi(kc,jc,im))*udzrq
               endif
            else
               dzzp = (phi(kc,jc,ip) - 2.0*phi(kc,jc,ic) + phi(kc,jc,im))*udzrq
            endif

                ! Extra nonlinear terms
                nlphi = pf_B*phi(kc,jc,ic)*(1.0 - phi(kc,jc,ic)) &
                        *(1.0 - 2.0*phi(kc,jc,ic) + pf_A*(tempr(kc,jc,ic) - pf_Tm))

                hphi(kc,jc,ic) = pf_D*(dyyp + dzzp) - nlphi
            end do
        end do
    end do

    if (salinity) then
        bcl = pf_B*pf_A*pf_Lambda
        do ic=xstartr(3),xendr(3)
            im=ic-1
            ip=ic+1
            do jc=xstartr(2),xendr(2)
                jm=jc-1
                jp=jc+1
                do kc=1,nxmr
                    hphi(kc,jc,ic) = hphi(kc,jc,ic) - &
                        bcl*sal(kc,jc,ic)*phi(kc,jc,ic)*(1.0 - phi(kc,jc,ic))
                end do
            end do
        end do
    end if

    return
end subroutine ExplicitTermsPhi
