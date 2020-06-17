/*
 * PlatoMaterialModelTest.cpp
 *
 *  Created on: Jun 11, 2020
 */

#include "PlatoTestHelpers.hpp"
#include <Teuchos_UnitTestHarness.hpp>

#include "MaterialModel.hpp"

namespace PlatoUnitTests
{


/******************************************************************************/
/*! 
  \brief Transform a block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_ScalarFunctor)
{

    // constructor tests
    //
    {
        Plato::ScalarFunctor tEmptyScalarFunctor;
        Plato::ScalarFunctor tConstantScalarFunctor(1.234);

        Plato::ScalarVector tResult("result", 2);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            tResult(aOrd) = tEmptyScalarFunctor(0.0);
            tResult(aOrd+1) = tConstantScalarFunctor(0.0);
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);
    
        TEST_ASSERT(tResult_Host(0) == 0.0);
        TEST_ASSERT(tResult_Host(1) == 1.234);
    }

    // linear functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tLinearScalarParams =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Specific Heat'>                  \n"
            "  <Parameter name='c0' type='double' value='900.0'/>  \n"
            "  <Parameter name='c1' type='double' value='5.0e-4'/> \n"
            "</ParameterList>                                      \n"
        );

        Plato::ScalarFunctor tLinearScalarFunctor(*tLinearScalarParams);
        Plato::ScalarVector tResult("result", 3);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            tResult(aOrd  ) = tLinearScalarFunctor(0.0);
            tResult(aOrd+1) = tLinearScalarFunctor(1000.0);
            tResult(aOrd+2) = tLinearScalarFunctor(1234.0);
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        TEST_ASSERT(tResult_Host(0) == 900.0 + 5.0e-4 * 0.0   );
        TEST_ASSERT(tResult_Host(1) == 900.0 + 5.0e-4 * 1000.0);
        TEST_ASSERT(tResult_Host(2) == 900.0 + 5.0e-4 * 1234.0);
    }

    // quadratic functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tQuadraticScalarParams =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Specific Heat'>                  \n"
            "  <Parameter name='c0' type='double' value='900.0'/>  \n"
            "  <Parameter name='c1' type='double' value='5.0e-4'/> \n"
            "  <Parameter name='c2' type='double' value='2.0e-7'/> \n"
            "</ParameterList>                                      \n"
        );

        Plato::ScalarFunctor tQuadraticScalarFunctor(*tQuadraticScalarParams);
        Plato::ScalarVector tResult("result", 4);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            Plato::Scalar tX[4] = {0.0, 1000.0, 1234.0, -1500.0};
            for (int i=0; i<4; i++)
            {
                tResult(aOrd+i) = tQuadraticScalarFunctor(tX[i]) - (900.0 + 5.0e-4 * tX[i] + 2.0e-7 * tX[i]*tX[i]);
            }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<4; i++)
        {
            TEST_ASSERT(tResult_Host(i) == 0);
        }
    }

    // quadratic functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tQuadraticScalarParams =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Specific Heat'>                  \n"
            "  <Parameter name='c0' type='double' value='900.0'/>  \n"
            "  <Parameter name='c1' type='double' value='0.0'/>    \n"
            "  <Parameter name='c2' type='double' value='2.0e-7'/> \n"
            "</ParameterList>                                      \n"
        );

        Plato::ScalarFunctor tQuadraticScalarFunctor(*tQuadraticScalarParams);
        Plato::ScalarVector tResult("result", 4);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            Plato::Scalar tX[4] = {0.0, 1000.0, 1234.0, -1500.0};
            for (int i=0; i<4; i++)
            {
                tResult(aOrd+i) = tQuadraticScalarFunctor(tX[i]) - (900.0 + 0.0 * tX[i] + 2.0e-7 * tX[i]*tX[i]);
            }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<4; i++)
        {
            TEST_ASSERT(tResult_Host(i) == 0);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Transform a block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_TensorFunctor)
{
    // zero tensor tests
    //
    {
        Plato::TensorFunctor<3> tEmptyTensorFunctor;
        std::vector<std::vector<Plato::Scalar>> tZeroTensor = {{0,0,0},{0,0,0},{0,0,0}};

        Plato::ScalarVector tResult("result", 1, 3, 3);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tResult(0, i, j) = tEmptyTensorFunctor(0.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);
    
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
            {
                TEST_ASSERT(tResult_Host(0, i, j) == tZeroTensor[i][j]);
            }
    }

    // constant diagonal tensor functor tests
    //
    {
        Plato::TensorFunctor<3> tDiagonalTensorFunctor(3.0);
        std::vector<std::vector<Plato::Scalar>> tDiagonalTensor = {{3,0,0},{0,3,0},{0,0,3}};

        Plato::ScalarArray3D tResult("result", 2, 3, 3);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tResult(0, i, j) = tDiagonalTensorFunctor(0.0, i, j);
                    tResult(1, i, j) = tDiagonalTensorFunctor(1.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);
    
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
            {
                TEST_ASSERT(tResult_Host(0, i, j) == tDiagonalTensor[i][j]);
                TEST_ASSERT(tResult_Host(1, i, j) == tDiagonalTensor[i][j]);
            }
    }

    // linear tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tLinearTensorParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Thermal Expansion'>                      \n"
                "  <Parameter name='c011' type='double' value='22.06e-6'/>     \n"
                "  <Parameter name='c111' type='double' value='3.9389e-8'/>    \n"
                "</ParameterList>                                              \n"
            );

        Plato::Scalar tC0 = 22.06e-6, tC1 = 3.9389e-8;

        Plato::ScalarArray3D tResult("result", 4, 3, 3);

        Plato::TensorFunctor<3> tLinearTensorFunctor(*tLinearTensorParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tResult(0, i, j) = tLinearTensorFunctor(0.0, i, j);
                    tResult(1, i, j) = tLinearTensorFunctor(1000.0, i, j);
                    tResult(2, i, j) = tLinearTensorFunctor(1234.0, i, j);
                    tResult(3, i, j) = tLinearTensorFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    if (i==j)
                    {
                        TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), tC0 + tC1*tValues[k], 1e-15);
                    } else {
                        TEST_ASSERT(tResult_Host(k, i, j) == 0.0);
                    }
                }
        }
    }

    // quadratic tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tLinearTensorParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Thermal Expansion'>                      \n"
                "  <Parameter name='c011' type='double' value='22.06e-6'/>     \n"
                "  <Parameter name='c111' type='double' value='3.9389e-8'/>    \n"
                "  <Parameter name='c211' type='double' value='-7.82412e-11'/> \n"
                "</ParameterList>                                              \n"
            );

        Plato::Scalar tC0 = 22.06e-6, tC1 = 3.9389e-8, tC2 = -7.82412e-11;

        Plato::ScalarArray3D tResult("result", 4, 3, 3);

        Plato::TensorFunctor<3> tLinearTensorFunctor(*tLinearTensorParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tResult(0, i, j) = tLinearTensorFunctor(0.0, i, j);
                    tResult(1, i, j) = tLinearTensorFunctor(1000.0, i, j);
                    tResult(2, i, j) = tLinearTensorFunctor(1234.0, i, j);
                    tResult(3, i, j) = tLinearTensorFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    if (i==j)
                    {
                        TEST_FLOATING_EQUALITY(tResult_Host(k, i, j),
                                               tC0 + tC1*tValues[k] + tC2*tValues[k]*tValues[k], 1e-15);
                    } else {
                        TEST_ASSERT(tResult_Host(k, i, j) == 0.0);
                    }
                }
        }
    }
}

} // namespace PlatoUnitTests
