#include <Teuchos_UnitTestHarness.hpp>

#include "PlatoTestHelpers.hpp"


namespace PlatoUtestHelpers
{

static Teuchos::RCP<Omega_h::Library> libOmegaH;

void finalizeOmegaH() {
  libOmegaH.reset();
}

void initializeOmegaH(int *argc , char ***argv)
{
  libOmegaH = Teuchos::rcp(new Omega_h::Library(argc, argv));
}

Teuchos::RCP<Omega_h::Library> getLibraryOmegaH()
{
  return libOmegaH;
}

void test_array_1d
(const Plato::ScalarVector& aInput,
 std::vector<Plato::Scalar>& aGold,
 Plato::Scalar tTol)
{
    auto tHostInput = Kokkos::create_mirror(aInput);
    Kokkos::deep_copy(tHostInput, aInput);

    const Plato::OrdinalType tDim0 = aInput.dimension_0();
    for(Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        //printf("X(%d) = %f\n", tIndexI, tHostInput(tIndexI));
        TEST_FLOATING_EQUALITY(tHostInput(tIndexI), aGold[tIndexI], tTol);
    }
}

void test_array_2d
(const Plato::ScalarMultiVector& aInput,
 std::vector<std::vector<Plato::Scalar>>& aGold,
 Plato::Scalar tTol)
{
    auto tHostInput = Kokkos::create_mirror(aInput);
    Kokkos::deep_copy(tHostInput, aInput);

    const Plato::OrdinalType tDim0 = aInput.dimension_0();
    const Plato::OrdinalType tDim1 = aInput.dimension_1();
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostInput(tIndexI, tIndexJ), aGold[tIndexI][tIndexJ], tTol);
        }
    }
}

void test_array_3d
(const Plato::ScalarArray3D& aInput,
 std::vector< std::vector< std::vector<Plato::Scalar> > >& aGold,
 Plato::Scalar tTol)
{
    auto tHostInput = Kokkos::create_mirror(aInput);
    Kokkos::deep_copy(tHostInput, aInput);

    const Plato::OrdinalType tDim0 = aInput.dimension_0();
    const Plato::OrdinalType tDim1 = aInput.dimension_1();
    const Plato::OrdinalType tDim2 = aInput.dimension_2();
    for(Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            for(Plato::OrdinalType tIndexK = 0; tIndexK < tDim2; tIndexK++)
            {
                //printf("X(%d,%d,%d) = %f\n", tIndexI, tIndexJ, tIndexK, tHostInput(tIndexI, tIndexJ, tIndexK));
                TEST_FLOATING_EQUALITY(tHostInput(tIndexI, tIndexJ, tIndexK), aGold[tIndexI][tIndexJ][tIndexK], tTol);
            }
        }
    }
}

} // namespace PlatoUtestHelpers
