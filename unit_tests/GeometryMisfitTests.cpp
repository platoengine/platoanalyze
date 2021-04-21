/*
 * GeometryMisfitTest.cpp
 *
 *  Created on: March 11, 2021
 */


#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "Geometrical.hpp"

#include <iostream>
#include <sstream>


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Misfit)
{
  // create geometry misfit criterion
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                  \n"
    "  <ParameterList name='Spatial Model'>                                                \n"
    "    <ParameterList name='Domains'>                                                    \n"
    "      <ParameterList name='Design Volume'>                                            \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                  \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>          \n"
    "      </ParameterList>                                                                \n"
    "    </ParameterList>                                                                  \n"
    "  </ParameterList>                                                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                   \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                           \n"
    "  <ParameterList name='Material Models'>                                              \n"
    "    <ParameterList name='Unobtainium'>                                                \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                 \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>                  \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                \n"
    "      </ParameterList>                                                                \n"
    "    </ParameterList>                                                                  \n"
    "  </ParameterList>                                                                    \n"
    "  <ParameterList name='Criteria'>                                                     \n"
    "    <ParameterList name='Geometry Misfit'>                                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function' />                 \n"
    "      <Parameter name='Linear' type='bool' value='true' />                            \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Geometry Misfit' /> \n"
    "      <Parameter name='Point Cloud File Name' type='string' value='points.xyz' />     \n"
    "      <Parameter name='Sides' type='string' value='x+' />                             \n"
    "    </ParameterList>                                                                  \n"
    "  </ParameterList>                                                                    \n"
    "</ParameterList>                                                                      \n"
  );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using GeometryT = typename Plato::Geometrical<tSpaceDim>;
    using ResidualT = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::Residual;

    std::string tFunctionName("Geometry Misfit");
    auto tGeometryMisfit = Plato::Geometric::GeometryMisfit<ResidualT>(tOnlyDomain, tDataMap, *tParamList, tFunctionName);

    auto aFileName = tParamList->sublist("Criteria").sublist("Geometry Misfit").get<std::string>("Point Cloud File Name");
    auto tHostPoints = tGeometryMisfit.readPointsFromFile(aFileName);

    // verify that points loaded correctly
    //
    std::vector<std::vector<double>> tPointData = {
      {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}
    };
    for (int iPoint=0; iPoint<tPointData.size(); iPoint++)
    {
        for (int iDim=0; iDim<tSpaceDim; iDim++)
        {
            TEST_FLOATING_EQUALITY(tPointData[iPoint][iDim], tHostPoints[iPoint][iDim], 1e-16);
        }
    }

    // check the DataMap
    //
    TEST_ASSERT(tDataMap.scalarMultiVectors.count(aFileName) == 1);
    auto tDevicePoints = tDataMap.scalarMultiVectors.at(aFileName);
    auto tDevicePoints_Host = Kokkos::create_mirror_view(tDevicePoints);
    Kokkos::deep_copy(tDevicePoints_Host, tDevicePoints);

    for (int iPoint=0; iPoint<tPointData.size(); iPoint++)
    {
        for (int iDim=0; iDim<tSpaceDim; iDim++)
        {
            TEST_FLOATING_EQUALITY(tPointData[iPoint][iDim], tDevicePoints_Host(iDim, iPoint), 1e-16);
        }
    }

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarMultiVectorT<ResidualT::ControlScalarType> tControlWS("design variables", tNumCells, GeometryT::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // evaluate the function
    //
    using WorksetBaseT = typename Plato::Geometric::WorksetBase<GeometryT>;
    WorksetBaseT tWorksetBase(*tMesh);

    {
        Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("configuration", tNumCells, GeometryT::mNumNodesPerCell, tSpaceDim);
        tWorksetBase.worksetConfig(tConfigWS, tOnlyDomain);

        Plato::ScalarVectorT<ResidualT::ResultScalarType> tResultWS("result", tNumCells);

        tGeometryMisfit.evaluate_boundary(tSpatialModel, tControlWS, tConfigWS, tResultWS);

        auto tResultWS_Host = Kokkos::create_mirror_view(tResultWS);
        Kokkos::deep_copy(tResultWS_Host, tResultWS);

        Plato::Scalar tError = 0.0;
        for (int iCell=0; iCell<tResultWS.extent(0); iCell++)
        {
            tError += tResultWS_Host(iCell);
        }
        TEST_FLOATING_EQUALITY(tError, 1.0, 1e-16);
    }


    {
        using GradientX = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::GradientX;
        auto tGeometryMisfit_GradientX = Plato::Geometric::GeometryMisfit<GradientX>(tOnlyDomain, tDataMap, *tParamList, tFunctionName);

        Plato::ScalarArray3DT<GradientX::ConfigScalarType> tConfigWS("configuration", tNumCells, GeometryT::mNumNodesPerCell, tSpaceDim);
        tWorksetBase.worksetConfig(tConfigWS, tOnlyDomain);

        Plato::ScalarVectorT<GradientX::ResultScalarType> tResultWS("result", tNumCells);

        tGeometryMisfit_GradientX.evaluate_boundary(tSpatialModel, tControlWS, tConfigWS, tResultWS);

        Plato::OrdinalType tNumNodesPerCell = tSpaceDim+1;
        Plato::OrdinalType tNumDofsPerCell = tSpaceDim*tNumNodesPerCell;
        Plato::ScalarMultiVector tResultWS_POD("result pod", tNumCells, tNumDofsPerCell);
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), LAMBDA_EXPRESSION(int aCellOrdinal)
        {
            for(Plato::OrdinalType iDof=0; iDof<tNumDofsPerCell; iDof++)
            {
                tResultWS_POD(aCellOrdinal, iDof) = tResultWS(aCellOrdinal).dx(iDof);
            }
        }, "flatten");


        auto tResultWS_Host = Kokkos::create_mirror_view(tResultWS_POD);
        Kokkos::deep_copy(tResultWS_Host, tResultWS_POD);

        std::vector<std::vector<Plato::Scalar>> tGold_gradX{
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, -0.5, 1, -0.5, 0.5, 0, 0.5, 0, 0, 0, 0},
          {0, -0.5, 0, 0, 0, 0.5, 1, 0.5, -0.5, 0, 0, 0}
        };

        for (int iCell=0; iCell<tResultWS.extent(0); iCell++)
        {
            for (int iDof=0; iDof<tNumDofsPerCell; iDof++)
            {
                if (tGold_gradX[iCell][iDof] == 0)
                {
                    TEST_ASSERT(fabs(tResultWS_Host(iCell, iDof)) < 1e-15);
                }
                else
                {
                    TEST_FLOATING_EQUALITY(tGold_gradX[iCell][iDof], tResultWS_Host(iCell, iDof), 1e-15);
                }
            }
        }
    }
}
