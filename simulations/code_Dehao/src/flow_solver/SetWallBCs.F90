!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                         ! 
!    FILE: SetTempBCs.F90                                 !
!    CONTAINS: subroutine SetTempBCs                      !
!                                                         ! 
!    PURPOSE: Initialization routine. Calcuates the       !
!     temperature boundary conditions at the plates       !
!                                                         !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine SetWallBCs
    use param
    use decomp_2d
    use local_arrays
    use mgrd_arrays, only: sal,phi
    use mpih
    implicit none


    vx(1,:,:)=0.0
    vx(nx,:,:)=0.0

    if(xstart(2).eq.1)then
            vy(:,1,:)=0.0
            temp(:,1,:)=temp(:,2,:)
    endif
    if(xend(2).eq.nym)then
            vy(:,ny,:)=0.0
            temp(:,nym,:)=temp(:,nym-1,:)
    endif
    if(xstart(3).eq.1)then
            vz(:,:,1)=0.0
            temp(:,:,1)=temp(:,:,2)
    endif
    if(xend(3).eq.nzm)then
            vz(:,:,nz)=0.0
            temp(:,:,nzm)=temp(:,:,nzm-1)
    endif

    if(salinity) then
    if(xstartr(2).eq.1)then
            sal(:,1,:)=sal(:,2,:)
    endif
    if(xendr(2).eq.nymr)then
            sal(:,nymr,:)=sal(:,nymr-1,:)
    endif
    if(xstartr(3).eq.1)then
            sal(:,:,1)=sal(:,:,2)
    endif
    if(xendr(3).eq.nzmr)then
            sal(:,:,nzmr)=sal(:,:,nzmr-1)
    endif
    endif

    return
end subroutine SetWallBCs
