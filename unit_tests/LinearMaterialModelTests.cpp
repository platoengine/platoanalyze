/*
 * LinearMaterialModelTests.cpp
 *
 *  Created on: Mar 23, 2020
 */

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "OrthotropicLinearElasticMaterial.hpp"

namespace OrthotropicLinearElasticMaterialTest
{

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic1D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                 \n"
      "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat(*tParams);
    auto tStiffMatrix = tOrthoMat.getStiffnessMatrix();
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(1.3461538, tStiffMatrix(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic1D_PoissonRatioKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Orthotropic Linear Elastic'>                 \n"
      "  <Parameter  name='Poissons Rtio' type='double' value='0.3'/>  \n"
      "  <Parameter  name='Youngs Modulus' type='double' value='1.0'/>   \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic1D_YoungModulusKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                 \n"
      "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Moulus' type='double' value='1.0'/>  \n"
      "    </ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='0.8'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat(*tParams);
    auto tStiffMatrix = tOrthoMat.getStiffnessMatrix();
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(1.0775862, tStiffMatrix(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.2586206, tStiffMatrix(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.2586206, tStiffMatrix(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.8620689, tStiffMatrix(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.3,       tStiffMatrix(2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_PoissonRatioKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissos Ratio XY' type='double' value='0.3'/> \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_ShearModulusKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulu XY' type='double' value='0.3'/>    \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusXKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulu X' type='double' value='1.0'/>    \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulu Y' type='double' value='1.0'/>    \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusXCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='-1.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='-1.0'/>  \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_PoissonRatioXYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.75'/> \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='2.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat(*tParams);
    auto tStiffMatrix = tOrthoMat.getStiffnessMatrix();
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(128.942, tStiffMatrix(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(5.253,   tStiffMatrix(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(5.253,   tStiffMatrix(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffMatrix(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffMatrix(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffMatrix(0,5), tTolerance);

    TEST_FLOATING_EQUALITY(5.253,  tStiffMatrix(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(13.309, tStiffMatrix(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(5.452,  tStiffMatrix(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(1,5), tTolerance);

    TEST_FLOATING_EQUALITY(5.253,  tStiffMatrix(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(5.452,  tStiffMatrix(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(13.309, tStiffMatrix(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(2,5), tTolerance);

    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(3.928, tStiffMatrix(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,5), tTolerance);

    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(6.6, tStiffMatrix(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,5), tTolerance);

    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(6.6, tStiffMatrix(5,5), tTolerance);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusXKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulu X' type='double' value='126.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "</ParameterList>                                                         \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulu Y' type='double' value='11.0'/>    \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulu Z' type='double' value='11.0'/>    \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusYZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulu YZ' type='double' value='3.928'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulu XZ' type='double' value='6.60'/>    \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulu XY' type='double' value='6.60'/>    \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonsRatioXYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Rati XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonsRatioXZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonsRatioYZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

// TODO: Stability Conditions - Unit Tests

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusXCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='-126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='-11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='-11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='-6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='-6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusYZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='-3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonRationXYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='3.40'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>  \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonRationXZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='3.40'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>  \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonRationYZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='2.0'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>  \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

}
// namespace OrthotropicLinearElasticMaterialTest


