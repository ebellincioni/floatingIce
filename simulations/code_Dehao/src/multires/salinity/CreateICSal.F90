!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         ! 
!    FILE: CreateICSal.F90                                !
!    CONTAINS: subroutine CreateICSal                     !
!                                                         ! 
!    PURPOSE: Initialization routine. Sets initial        !
!     conditions for salinity field                       !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine CreateICSal
    use param
    use mgrd_arrays, only: sal,phi
    use decomp_2d, only: xstartr,xendr
    use mpih
    implicit none
    integer :: i,k,j, kmid
    real :: xxx,yyy,eps,varptb,amp,r
    real :: gamma, t0, x0, h0, A, B, alpha
    real, dimension(11) :: yh, zh

        do i=xstartr(3),xendr(3)
            do j=xstartr(2),xendr(2)
                do k=1,nxmr
                    r = sqrt((xmr(k) - alx3*0.5)**2 + (ymr(j) - ylen*0.5)**2)
                    sal(k,j,i) = 0.5*(1.0 + tanh((r - 0.05)/2/pf_eps))!0.5*(1.0 - tanh((xmr(k) - 0.9)/2/pf_eps))
!                    sal(k,j,i) = sal(k,j,i) - 5.0/7.5*(ymr(j)/ylen)*(1.0-phi(k,j,i))
                end do
            end do
        end do




    
    return
end subroutine CreateICSal
