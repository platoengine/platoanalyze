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
  \brief Unit tests for Plato::ScalarFunctor class
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
  \brief Unit tests for Plato::TensorConstant class
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_TensorConstant)
{
    // zero tensor tests
    //
    {
        Plato::TensorConstant<3> tEmptyTensorConstant;
        std::vector<std::vector<Plato::Scalar>> tZeroTensor = {{0,0,0},{0,0,0},{0,0,0}};

        Plato::ScalarArray3D tResult("result", 1, 3, 3);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tResult(0, i, j) = tEmptyTensorConstant(i, j);
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

    // diagonal tensor constant tests
    //
    {
        Plato::TensorConstant<3> tDiagonalTensorConstant(3.0);
        std::vector<std::vector<Plato::Scalar>> tDiagonalTensor = {{3,0,0},{0,3,0},{0,0,3}};

        Plato::ScalarArray3D tResult("result", 1, 3, 3);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tResult(0, i, j) = tDiagonalTensorConstant(i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
            {
                TEST_ASSERT(tResult_Host(0, i, j) == tDiagonalTensor[i][j]);
            }
    }

}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::TensorFunctor class
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_TensorFunctor)
{
    // zero tensor tests
    //
    {
        Plato::TensorFunctor<3> tEmptyTensorFunctor;
        std::vector<std::vector<Plato::Scalar>> tZeroTensor = {{0,0,0},{0,0,0},{0,0,0}};

        Plato::ScalarArray3D tResult("result", 1, 3, 3);

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

/******************************************************************************/
/*!
  \brief Unit tests for Plato::Rank4VoigtFunctor class
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_Rank4VoigtFunctor)
{
    std::vector<std::vector<Plato::Scalar>>
      tRank4Voigt_c0 = {
        {100,  80,  80,  0,  0,  0},
        { 80, 100,  80,  0,  0,  0},
        { 80,  80, 100,  0,  0,  0},
        {  0,   0,   0, 90,  0,  0},
        {  0,   0,   0,  0, 90,  0},
        {  0,   0,   0,  0,  0, 90}
      };

    std::vector<std::vector<Plato::Scalar>>
      tRank4Voigt_c1 = {
        {2.0, 1.0, 1.0,   0,   0,   0},
        {1.0, 2.0, 1.0,   0,   0,   0},
        {1.0, 1.0, 2.0,   0,   0,   0},
        {  0,   0,   0, 1.5,   0,   0},
        {  0,   0,   0,   0, 1.5,   0},
        {  0,   0,   0,   0,   0, 1.5}
      };

    std::vector<std::vector<Plato::Scalar>>
      tRank4Voigt_c2 = {
        {2.0e-2, 1.0e-2, 1.0e-2,      0,      0,      0},
        {1.0e-2, 2.0e-2, 1.0e-2,      0,      0,      0},
        {1.0e-2, 1.0e-2, 2.0e-2,      0,      0,      0},
        {     0,      0,      0, 1.5e-2,      0,      0},
        {     0,      0,      0,      0, 1.5e-2,      0},
        {     0,      0,      0,      0,      0, 1.5e-2}
      };

    // zero rank4voigt tensor functor tests
    //
    {
        Plato::Rank4VoigtFunctor<3> tEmptyRank4VoigtFunctor;
        std::vector<std::vector<Plato::Scalar>>
          tZeroTensor = {
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0}
          };

        Plato::ScalarArray3D tResult("result", 1, 6, 6);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tEmptyRank4VoigtFunctor(0.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<6; i++)
            for (int j=0; j<6; j++)
            {
                TEST_ASSERT(tResult_Host(0, i, j) == tZeroTensor[i][j]);
            }
    }

    // constant rank4voigt tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tConstantRank4VoigtParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>               \n"
                "  <Parameter name='c011' type='double' value='100.0'/> \n"
                "  <Parameter name='c012' type='double' value='80.0'/>  \n"
                "  <Parameter name='c013' type='double' value='80.0'/>  \n"
                "  <Parameter name='c044' type='double' value='90.0'/>  \n"
                "</ParameterList>                                       \n"
            );

        Plato::ScalarArray3D tResult("result", 4, 6, 6);

        Plato::Rank4VoigtFunctor<3> tConstantRank4VoigtFunctor(*tConstantRank4VoigtParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tConstantRank4VoigtFunctor(    0.0, i, j);
                    tResult(1, i, j) = tConstantRank4VoigtFunctor( 1000.0, i, j);
                    tResult(2, i, j) = tConstantRank4VoigtFunctor( 1234.0, i, j);
                    tResult(3, i, j) = tConstantRank4VoigtFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), tRank4Voigt_c0[i][j], 1e-15);
                }
        }
    }

    // linear rank4voigt tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tLinearRank4VoigtParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>               \n"
                "  <Parameter name='c011' type='double' value='100.0'/> \n"
                "  <Parameter name='c012' type='double' value='80.0'/>  \n"
                "  <Parameter name='c013' type='double' value='80.0'/>  \n"
                "  <Parameter name='c044' type='double' value='90.0'/>  \n"
                "  <Parameter name='c111' type='double' value='2.0'/>   \n"
                "  <Parameter name='c112' type='double' value='1.0'/>   \n"
                "  <Parameter name='c113' type='double' value='1.0'/>   \n"
                "  <Parameter name='c144' type='double' value='1.5'/>   \n"
                "</ParameterList>                                       \n"
            );

        Plato::ScalarArray3D tResult("result", 4, 6, 6);

        Plato::Rank4VoigtFunctor<3> tLinearRank4VoigtFunctor(*tLinearRank4VoigtParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tLinearRank4VoigtFunctor(    0.0, i, j);
                    tResult(1, i, j) = tLinearRank4VoigtFunctor( 1000.0, i, j);
                    tResult(2, i, j) = tLinearRank4VoigtFunctor( 1234.0, i, j);
                    tResult(3, i, j) = tLinearRank4VoigtFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    auto x = tValues[k];
                    auto c0 = tRank4Voigt_c0[i][j];
                    auto c1 = tRank4Voigt_c1[i][j];
                    TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), c0 + c1*x, 1e-15);
                }
        }
    }

    // quadratic rank4voigt tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tQuadraticRank4VoigtParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>                \n"
                "  <Parameter name='c011' type='double' value='100.0'/>  \n"
                "  <Parameter name='c012' type='double' value='80.0'/>   \n"
                "  <Parameter name='c013' type='double' value='80.0'/>   \n"
                "  <Parameter name='c044' type='double' value='90.0'/>   \n"
                "  <Parameter name='c111' type='double' value='2.0'/>    \n"
                "  <Parameter name='c112' type='double' value='1.0'/>    \n"
                "  <Parameter name='c113' type='double' value='1.0'/>    \n"
                "  <Parameter name='c144' type='double' value='1.5'/>    \n"
                "  <Parameter name='c211' type='double' value='2.0e-2'/> \n"
                "  <Parameter name='c212' type='double' value='1.0e-2'/> \n"
                "  <Parameter name='c213' type='double' value='1.0e-2'/> \n"
                "  <Parameter name='c244' type='double' value='1.5e-2'/> \n"
                "</ParameterList>                                        \n"
            );

        Plato::ScalarArray3D tResult("result", 4, 6, 6);

        Plato::Rank4VoigtFunctor<3> tQuadraticRank4VoigtFunctor(*tQuadraticRank4VoigtParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tQuadraticRank4VoigtFunctor(    0.0, i, j);
                    tResult(1, i, j) = tQuadraticRank4VoigtFunctor( 1000.0, i, j);
                    tResult(2, i, j) = tQuadraticRank4VoigtFunctor( 1234.0, i, j);
                    tResult(3, i, j) = tQuadraticRank4VoigtFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    auto x = tValues[k];
                    auto c0 = tRank4Voigt_c0[i][j];
                    auto c1 = tRank4Voigt_c1[i][j];
                    auto c2 = tRank4Voigt_c2[i][j];
                    TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), c0 + c1*x + c2*x*x, 1e-15);
                }
        }
    }
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::Rank4VoigtConstant class
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_Rank4VoigtConstant)
{
    // zero rank4voigt tensor constant tests
    //
    {
        Plato::Rank4VoigtConstant<3> tEmptyRank4VoigtConstant;
        std::vector<std::vector<Plato::Scalar>>
          tZeroTensor = {
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0},
            {0,0,0,0,0,0}
          };

        Plato::ScalarArray3D tResult("result", 1, 6, 6);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tEmptyRank4VoigtConstant(i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<6; i++)
            for (int j=0; j<6; j++)
            {
                TEST_ASSERT(tResult_Host(0, i, j) == tZeroTensor[i][j]);
            }
    }

    // constant rank4voigt tensor constant tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tConstantRank4VoigtParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>              \n"
                "  <Parameter name='c11' type='double' value='100.0'/> \n"
                "  <Parameter name='c12' type='double' value='80.0'/>  \n"
                "  <Parameter name='c13' type='double' value='80.0'/>  \n"
                "  <Parameter name='c44' type='double' value='90.0'/>  \n"
                "</ParameterList>                                      \n"
            );

        Plato::ScalarArray3D tResult("result", 1, 6, 6);

        std::vector<std::vector<Plato::Scalar>>
          tConstantRank4Voigt = {
            {100,  80,  80,  0,  0,  0},
            { 80, 100,  80,  0,  0,  0},
            { 80,  80, 100,  0,  0,  0},
            {  0,   0,   0, 90,  0,  0},
            {  0,   0,   0,  0, 90,  0},
            {  0,   0,   0,  0,  0, 90}
          };

        Plato::Rank4VoigtConstant<3> tRank4VoigtConstant(*tConstantRank4VoigtParams);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tRank4VoigtConstant(i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<6; i++)
            for (int j=0; j<6; j++)
            {
                TEST_FLOATING_EQUALITY(tResult_Host(0, i, j), tConstantRank4Voigt[i][j], 1e-15);
            }
    }
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::IsotropicStiffnessFunctor class
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_IsotropicStiffnessFunctor)
{
    Plato::Scalar v = 0.35, E0 = 6.90342e10, E1 = -3.33342e7, E2 = -1.26564e4;
    Plato::Scalar c = 1.0/((1.0+v)*(1.0-2.0*v));
    Plato::Scalar c000 = E0*c*(1.0-v), c001 = E0*c*v, c033 = 1.0/2.0*E0*c*(1.0-2.0*v);
    Plato::Scalar c100 = E1*c*(1.0-v), c101 = E1*c*v, c133 = 1.0/2.0*E1*c*(1.0-2.0*v);
    Plato::Scalar c200 = E2*c*(1.0-v), c201 = E2*c*v, c233 = 1.0/2.0*E2*c*(1.0-2.0*v);

    std::vector<std::vector<Plato::Scalar>>
      tIsotropic_c0 = {
        {c000, c001, c001,    0,    0,    0},
        {c001, c000, c001,    0,    0,    0},
        {c001, c001, c000,    0,    0,    0},
        {   0,    0,    0, c033,    0,    0},
        {   0,    0,    0,    0, c033,    0},
        {   0,    0,    0,    0,    0, c033}
      };

    std::vector<std::vector<Plato::Scalar>>
      tIsotropic_c1 = {
        {c100, c101, c101,    0,    0,    0},
        {c101, c100, c101,    0,    0,    0},
        {c101, c101, c100,    0,    0,    0},
        {   0,    0,    0, c133,    0,    0},
        {   0,    0,    0,    0, c133,    0},
        {   0,    0,    0,    0,    0, c133}
      };

    std::vector<std::vector<Plato::Scalar>>
      tIsotropic_c2 = {
        {c200, c201, c201,    0,    0,    0},
        {c201, c200, c201,    0,    0,    0},
        {c201, c201, c200,    0,    0,    0},
        {   0,    0,    0, c233,    0,    0},
        {   0,    0,    0,    0, c233,    0},
        {   0,    0,    0,    0,    0, c233}
      };

    // constant isotropic stiffness voigt tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tConstantIsotropicParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>                        \n"
                "  <ParameterList name='Youngs Modulus'>                         \n"
                "    <Parameter name='c0' type='double' value='6.90342e10'/>     \n"
                "  </ParameterList>                                              \n"
                "  <Parameter name='Poissons Ratio' type='double' value='0.35'/> \n"
                "</ParameterList>                                                \n"
            );

        Plato::ScalarArray3D tResult("result", 4, 6, 6);

        Plato::IsotropicStiffnessFunctor<3> tConstantIsotropicFunctor(*tConstantIsotropicParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tConstantIsotropicFunctor(    0.0, i, j);
                    tResult(1, i, j) = tConstantIsotropicFunctor( 1000.0, i, j);
                    tResult(2, i, j) = tConstantIsotropicFunctor( 1234.0, i, j);
                    tResult(3, i, j) = tConstantIsotropicFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), tIsotropic_c0[i][j], 1e-15);
                }
        }
    }


    // linear isotropic stiffness voigt tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tLinearIsotropicParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>                        \n"
                "  <ParameterList name='Youngs Modulus'>                         \n"
                "    <Parameter name='c0' type='double' value='6.90342e10'/>     \n"
                "    <Parameter name='c1' type='double' value='-3.33342e7'/>     \n"
                "  </ParameterList>                                              \n"
                "  <Parameter name='Poissons Ratio' type='double' value='0.35'/> \n"
                "</ParameterList>                                                \n"
            );

        Plato::ScalarArray3D tResult("result", 4, 6, 6);

        Plato::IsotropicStiffnessFunctor<3> tLinearIsotropicFunctor(*tLinearIsotropicParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tLinearIsotropicFunctor(    0.0, i, j);
                    tResult(1, i, j) = tLinearIsotropicFunctor( 1000.0, i, j);
                    tResult(2, i, j) = tLinearIsotropicFunctor( 1234.0, i, j);
                    tResult(3, i, j) = tLinearIsotropicFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    auto x = tValues[k];
                    auto c0 = tIsotropic_c0[i][j];
                    auto c1 = tIsotropic_c1[i][j];
                    TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), c0 + c1*x, 1e-15);
                }
        }
    }

    // quadratic isotropic stiffness voigt tensor functor tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tQuadraticIsotropicParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>                        \n"
                "  <ParameterList name='Youngs Modulus'>                         \n"
                "    <Parameter name='c0' type='double' value='6.90342e10'/>     \n"
                "    <Parameter name='c1' type='double' value='-3.33342e7'/>     \n"
                "    <Parameter name='c2' type='double' value='-1.26564e4'/>     \n"
                "  </ParameterList>                                              \n"
                "  <Parameter name='Poissons Ratio' type='double' value='0.35'/> \n"
                "</ParameterList>                                                \n"
            );

        Plato::ScalarArray3D tResult("result", 4, 6, 6);

        Plato::IsotropicStiffnessFunctor<3> tQuadraticIsotropicFunctor(*tQuadraticIsotropicParams);
        std::vector<Plato::Scalar> tValues = {0.0, 1000.0, 1234.0, -1500.0};
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tQuadraticIsotropicFunctor(    0.0, i, j);
                    tResult(1, i, j) = tQuadraticIsotropicFunctor( 1000.0, i, j);
                    tResult(2, i, j) = tQuadraticIsotropicFunctor( 1234.0, i, j);
                    tResult(3, i, j) = tQuadraticIsotropicFunctor(-1500.0, i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int k=0; k<tValues.size(); k++)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    auto x = tValues[k];
                    auto c0 = tIsotropic_c0[i][j];
                    auto c1 = tIsotropic_c1[i][j];
                    auto c2 = tIsotropic_c2[i][j];
                    TEST_FLOATING_EQUALITY(tResult_Host(k, i, j), c0 + c1*x + c2*x*x, 1e-15);
                }
        }
    }
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::IsotropicStiffnessConstant class
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_IsotropicStiffnessConstant)
{
    Plato::Scalar v = 0.35, E0 = 6.90342e10;
    Plato::Scalar c = 1.0/((1.0+v)*(1.0-2.0*v));
    Plato::Scalar c00 = E0*c*(1.0-v), c01 = E0*c*v, c33 = 1.0/2.0*E0*c*(1.0-2.0*v);

    std::vector<std::vector<Plato::Scalar>>
      tIsotropic_c0 = {
        {c00, c01, c01,   0,   0,   0},
        {c01, c00, c01,   0,   0,   0},
        {c01, c01, c00,   0,   0,   0},
        {  0,   0,   0, c33,   0,   0},
        {  0,   0,   0,   0, c33,   0},
        {  0,   0,   0,   0,   0, c33}
      };

    // constant isotropic stiffness voigt tensor constant tests
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tConstantIsotropicParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Elastic Stiffness'>                              \n"
                "  <Parameter name='Youngs Modulus' type='double' value='6.90342e10'/> \n"
                "  <Parameter name='Poissons Ratio' type='double' value='0.35'/>       \n"
                "</ParameterList>                                                      \n"
            );

        Plato::ScalarArray3D tResult("result", 1, 6, 6);

        Plato::IsotropicStiffnessConstant<3> tIsotropicConstant(*tConstantIsotropicParams);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tResult(0, i, j) = tIsotropicConstant(i, j);
                }
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        for (int i=0; i<6; i++)
            for (int j=0; j<6; j++)
            {
                TEST_FLOATING_EQUALITY(tResult_Host(0, i, j), tIsotropic_c0[i][j], 1e-15);
            }
    }
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::MaterialModel base
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMaterialModel_MaterialModel)
{
    // create new with empty parameter list.  type() should return Linear.
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Material Model'> \n"
                "</ParameterList>                      \n"
            );
        Plato::MaterialModel<3> tModel(*tParams);
        TEST_ASSERT(tModel.type() == Plato::MaterialModelType::Linear);
    }

    // create new with "Temperature Dependent" set to true in parameter list.
    // type() should return Nonlinear.
    //
    {
        Teuchos::RCP<Teuchos::ParameterList> tParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Material Model'>                                \n"
                "  <Parameter name='Temperature Dependent' type='bool' value='true'/> \n"
                "</ParameterList>                                                     \n"
            );
        Plato::MaterialModel<3> tModel(*tParams);
        TEST_ASSERT(tModel.type() == Plato::MaterialModelType::Nonlinear);
    }

    // create new with "Temperature Dependent" set to true.  Parse a scalar
    // functor, get it, and test it
    // tested function:  Plato::MaterialModel::parseScalar
    {
        Teuchos::RCP<Teuchos::ParameterList> tParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Material Model'>                                \n"
                "  <Parameter name='Temperature Dependent' type='bool' value='true'/> \n"
                "  <Parameter name='Some Scalar' type='double' value='1.234'/>        \n"
                "</ParameterList>                                                     \n"
            );
        Plato::MaterialModel<3> tModel(*tParams);
        tModel.parseScalar("Some Scalar", *tParams);
        auto tFunctor = tModel.getScalarFunctor("Some Scalar");
        Plato::ScalarVector tResult("result", 1);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            tResult(aOrd) = tFunctor(0.0);
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        TEST_ASSERT(tResult_Host(0) == 1.234);
    }

    // create new with "Temperature Dependent" set to false.
    // Parse a scalar constant, get it, and test it
    // tested function:  Plato::MaterialModel::parseScalar
    {
        Teuchos::RCP<Teuchos::ParameterList> tParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Material Model'>                         \n"
                "  <Parameter name='Some Scalar' type='double' value='1.234'/> \n"
                "</ParameterList>                                              \n"
            );
        Plato::MaterialModel<3> tModel(*tParams);
        tModel.parseScalar("Some Scalar", *tParams);
        auto tConstant = tModel.getScalarConstant("Some Scalar");
        Plato::ScalarVector tResult("result", 1);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            tResult(aOrd) = tConstant;
        }, "eval");
        auto tResult_Host = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tResult_Host, tResult);

        TEST_ASSERT(tResult_Host(0) == 1.234);
    }

    // create new with "Temperature Dependent" set to false.
    // Parse a scalar constant that doesn't exist but provide a default, get it, and test it
    // Parse a scalar constant that does exist, get it, and test it
    // Parse a tensor constant from a ParameterList with one term, get it, and test it
    // Parse a tensor constant from a ParameterList with two terms, get it, and test it
    // Parse a tensor constant from a Parameter, get it, and test it
    // tested function:  Plato::MaterialModel::parseScalarConstant
    // tested function:  Plato::MaterialModel::parseTensor
    // tested function:  Plato::MaterialModel::parseRank4Voigt
    {
        Teuchos::RCP<Teuchos::ParameterList> tParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Material Model'>                               \n"
                "  <Parameter name='Defined Scalar' type='double' value='1.234'/>    \n"
                "  <ParameterList name='Tensor Constant 1'>                          \n"
                "    <Parameter name='c11' type='double' value='2.345'/>             \n"
                "  </ParameterList>                                                  \n"
                "  <ParameterList name='Tensor Constant 2'>                          \n"
                "    <Parameter name='c11' type='double' value='2.345'/>             \n"
                "    <Parameter name='c22' type='double' value='3.456'/>             \n"
                "  </ParameterList>                                                  \n"
                "  <Parameter name='Tensor Constant 3' type='double' value='4.567'/> \n"
                "  <ParameterList name='Rank4Voigt Constant'>                        \n"
                "    <Parameter name='c11' type='double' value='100.0'/>             \n"
                "    <Parameter name='c12' type='double' value='80.0'/>              \n"
                "    <Parameter name='c13' type='double' value='80.0'/>              \n"
                "    <Parameter name='c44' type='double' value='90.0'/>              \n"
                "  </ParameterList>                                                  \n"
                "</ParameterList>                                                    \n"
            );
        Plato::MaterialModel<3> tLinearModel(*tParams);
        tLinearModel.parseScalarConstant("Some Scalar", *tParams, -1.234);
        tLinearModel.parseScalarConstant("Defined Scalar", *tParams);
        tLinearModel.parseTensor("Tensor Constant 1", *tParams);
        tLinearModel.parseTensor("Tensor Constant 2", *tParams);
        tLinearModel.parseTensor("Tensor Constant 3", *tParams);
        tLinearModel.parseRank4Voigt("Rank4Voigt Constant", *tParams);
        auto tDefaultConstant = tLinearModel.getScalarConstant("Some Scalar");
        auto tDefinedConstant = tLinearModel.getScalarConstant("Defined Scalar");
        auto tTensorConstant1 = tLinearModel.getTensorConstant("Tensor Constant 1");
        auto tTensorConstant2 = tLinearModel.getTensorConstant("Tensor Constant 2");
        auto tTensorConstant3 = tLinearModel.getTensorConstant("Tensor Constant 3");
        auto tRank4VoigtConstant = tLinearModel.getRank4VoigtConstant("Rank4Voigt Constant");

        Plato::ScalarVector tScalarResult("result", 2);
        Plato::ScalarArray3D tTensorResult("result", 3, 3, 3);
        Plato::ScalarArray3D tRank4VoigtResult("result", 3, 6, 6);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            tScalarResult(0) = tDefaultConstant;
            tScalarResult(1) = tDefinedConstant;

            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tTensorResult(0, i, j) = tTensorConstant1(i, j);
                    tTensorResult(1, i, j) = tTensorConstant2(i, j);
                    tTensorResult(2, i, j) = tTensorConstant3(i, j);
                }

            for (int i=0; i<6; i++)
                for (int j=0; j<6; j++)
                {
                    tRank4VoigtResult(0, i, j) = tRank4VoigtConstant(i, j);
                }
        }, "eval");
        auto tScalarResult_Host = Kokkos::create_mirror_view(tScalarResult);
        Kokkos::deep_copy(tScalarResult_Host, tScalarResult);
        auto tTensorResult_Host = Kokkos::create_mirror_view(tTensorResult);
        Kokkos::deep_copy(tTensorResult_Host, tTensorResult);
        auto tRank4VoigtResult_Host = Kokkos::create_mirror_view(tRank4VoigtResult);
        Kokkos::deep_copy(tRank4VoigtResult_Host, tRank4VoigtResult);

        TEST_ASSERT(tScalarResult_Host(0) == -1.234);
        TEST_ASSERT(tScalarResult_Host(1) ==  1.234);

        std::vector<std::vector<Plato::Scalar>> tTensorGold1 = {{2.345,0,0},{0,2.345,0},{0,0,2.345}};
        std::vector<std::vector<Plato::Scalar>> tTensorGold2 = {{2.345,0,0},{0,3.456,0},{0,0,2.345}};
        std::vector<std::vector<Plato::Scalar>> tTensorGold3 = {{4.567,0,0},{0,4.567,0},{0,0,4.567}};

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
            {
                TEST_ASSERT(tTensorResult_Host(0, i, j) == tTensorGold1[i][j]);
                TEST_ASSERT(tTensorResult_Host(1, i, j) == tTensorGold2[i][j]);
                TEST_ASSERT(tTensorResult_Host(2, i, j) == tTensorGold3[i][j]);
            }

        std::vector<std::vector<Plato::Scalar>>
        tRank4VoigtGold = {
            {100,  80,  80,  0,  0,  0},
            { 80, 100,  80,  0,  0,  0},
            { 80,  80, 100,  0,  0,  0},
            {  0,   0,   0, 90,  0,  0},
            {  0,   0,   0,  0, 90,  0},
            {  0,   0,   0,  0,  0, 90}
        };

        for (int i=0; i<6; i++)
            for (int j=0; j<6; j++)
            {
                TEST_ASSERT(tRank4VoigtResult_Host(0, i, j) == tRank4VoigtGold[i][j]);
            }
    }

    // create new with "Temperature Dependent" set to true.
    // Parse a scalar functor, get it, and test it
    // Parse a tensor functor from a ParameterList with one term, get it, and test it
    // Parse a tensor functor from a ParameterList with two terms, get it, and test it
    // Parse a tensor functor from a Parameter, get it, and test it
    // tested function:  Plato::MaterialModel::parseScalar
    // tested function:  Plato::MaterialModel::parseTensor
    {
        Teuchos::RCP<Teuchos::ParameterList> tParams =
            Teuchos::getParametersFromXmlString(
                "<ParameterList name='Material Model'>                                \n"
                "  <Parameter name='Temperature Dependent' type='bool' value='true'/> \n"
                "  <Parameter name='Scalar' type='double' value='1.234'/>             \n"
                "  <ParameterList name='Tensor Functor 1'>                            \n"
                "    <Parameter name='c011' type='double' value='2.345'/>             \n"
                "  </ParameterList>                                                   \n"
                "  <ParameterList name='Tensor Functor 2'>                            \n"
                "    <Parameter name='c011' type='double' value='2.345'/>             \n"
                "    <Parameter name='c022' type='double' value='3.456'/>             \n"
                "  </ParameterList>                                                   \n"
                "  <Parameter name='Tensor Functor 3' type='double' value='4.567'/>   \n"
                "</ParameterList>                                                     \n"
            );
        Plato::MaterialModel<3> tNonlinearModel(*tParams);
        tNonlinearModel.parseScalar("Scalar", *tParams);
        tNonlinearModel.parseTensor("Tensor Functor 1", *tParams);
        tNonlinearModel.parseTensor("Tensor Functor 2", *tParams);
        tNonlinearModel.parseTensor("Tensor Functor 3", *tParams);
        auto tScalarFunctor = tNonlinearModel.getScalarFunctor("Scalar");
        auto tTensorFunctor1 = tNonlinearModel.getTensorFunctor("Tensor Functor 1");
        auto tTensorFunctor2 = tNonlinearModel.getTensorFunctor("Tensor Functor 2");
        auto tTensorFunctor3 = tNonlinearModel.getTensorFunctor("Tensor Functor 3");

        Plato::ScalarVector tScalarResult("result", 1);
        Plato::ScalarArray3D tTensorResult("result", 3, 3, 3);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,1), LAMBDA_EXPRESSION(const int aOrd)
        {
            tScalarResult(0) = tScalarFunctor(0.0);

            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                {
                    tTensorResult(0, i, j) = tTensorFunctor1(0.0, i, j);
                    tTensorResult(1, i, j) = tTensorFunctor2(0.0, i, j);
                    tTensorResult(2, i, j) = tTensorFunctor3(0.0, i, j);
                }
        }, "eval");
        auto tScalarResult_Host = Kokkos::create_mirror_view(tScalarResult);
        Kokkos::deep_copy(tScalarResult_Host, tScalarResult);
        auto tTensorResult_Host = Kokkos::create_mirror_view(tTensorResult);
        Kokkos::deep_copy(tTensorResult_Host, tTensorResult);

        TEST_ASSERT(tScalarResult_Host(0) == 1.234);

        std::vector<std::vector<Plato::Scalar>> tTensorGold1 = {{2.345,0,0},{0,2.345,0},{0,0,2.345}};
        std::vector<std::vector<Plato::Scalar>> tTensorGold2 = {{2.345,0,0},{0,3.456,0},{0,0,2.345}};
        std::vector<std::vector<Plato::Scalar>> tTensorGold3 = {{4.567,0,0},{0,4.567,0},{0,0,4.567}};

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
            {
                TEST_ASSERT(tTensorResult_Host(0, i, j) == tTensorGold1[i][j]);
                TEST_ASSERT(tTensorResult_Host(1, i, j) == tTensorGold2[i][j]);
                TEST_ASSERT(tTensorResult_Host(2, i, j) == tTensorGold3[i][j]);
            }
    }
}


} // namespace PlatoUnitTests
