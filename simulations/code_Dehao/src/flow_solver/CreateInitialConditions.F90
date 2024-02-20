!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         ! 
!    FILE: CreateInitialConditions.F90                    !
!    CONTAINS: subroutine CreateInitialConditions         !
!                                                         ! 
!    PURPOSE: Initialization routine. Sets initial        !
!     conditions for velocity and temperature             !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine CreateInitialConditions
    use param
    use local_arrays, only: vy,vx,temp,vz
    use decomp_2d, only: xstart,xend
    use mpih
    implicit none
    integer :: j,k,i,kmid, io
    real :: xxx,yyy,zzz,eps,varptb,amp
    real :: h0,t0,Lambda,r, x0, A, B, alpha
    real, dimension(11) :: yh, zh
    logical :: exists


    do i=xstart(3),xend(3)
       do j=xstart(2),xend(2)
          do k=1,nxm
             vx(k,j,i) = 0.0d0
             vy(k,j,i) = 0.0d0
             vz(k,j,i) = 0.0d0
             r = sqrt((xm(k) - alx3*0.5)**2 + (ym(j) - ylen*0.5)**2)
             temp(k,j,i) = 0.5*(1.0 + tanh((r - 0.05)/2/pf_eps))  
            end do
       end do
    end do

    return
end subroutine CreateInitialConditions
