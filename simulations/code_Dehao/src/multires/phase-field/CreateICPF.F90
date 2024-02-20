!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         !
!    FILE: CreateICPF.F90                                 !
!    CONTAINS: subroutine CreateICPF                      !
!                                                         !
!    PURPOSE: Initialization routine. Sets initial        !
!     conditions for phase-field                          !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine CreateICPF

    use param
    use mgrd_arrays, only: phi
    use decomp_2d, only: xstartr,xendr
    use mpih

    implicit none

    integer :: i,j,k,kmid
    real :: r, x0, lambda, h0, t0, A, B, alpha, cylRad
    real, dimension(11) :: yh, zh

    
        do i=xstartr(3),xendr(3)
            do j=xstartr(2),xendr(2)
                do k=1,nxmr
                    cylRad = 0.05
                    r = sqrt((xmr(k) - (alx3-cylRad))**2 + (ymr(j) - ylen*0.5)**2)
                    phi(k,j,i) = 0.5*(1.0 - tanh((r - cylRad)/2/pf_eps))
                end do
            end do
        end do

    return

end subroutine CreateICPF

subroutine read_phase_field_params(A, B, alpha)
    implicit none
    real, intent(out) :: A, B, alpha

    integer :: io
    logical :: exists

    inquire(file="pfparam.in", exist=exists)
    if (exists) then
        open(newunit=io, file="pfparam.in", status="old", action="read")
        read(io, *) A, B, alpha
        close(io)
    else
        A = 1.132
        B = 0.3796
        alpha = 3.987e-2
    end if
end subroutine read_phase_field_params
