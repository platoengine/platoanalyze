/*
 * CubicLinearElasticMaterial.cpp
 *
 *  Created on: Mar 24, 2020
 */

#include "CubicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Linear elastic cubic material model constructor. - 1D
**********************************************************************************/
template<>
Plato::CubicLinearElasticMaterial<1>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
    Plato::LinearElasticMaterial<1>(paramList)
{
    Plato::Scalar tC11 = paramList.get<Plato::Scalar>("C11");
    mCellStiffness(0, 0) = tC11;
}

/******************************************************************************//**
 * \brief Linear elastic cubic material model constructor. - 2D
**********************************************************************************/
template<>
Plato::CubicLinearElasticMaterial<2>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
    Plato::LinearElasticMaterial<2>(paramList)
{
    Plato::Scalar tC11 = paramList.get<Plato::Scalar>("C11");
    Plato::Scalar tC12 = paramList.get<Plato::Scalar>("C12");
    Plato::Scalar tC44 = paramList.get<Plato::Scalar>("C44");

    mCellStiffness(0, 0) = tC11;
    mCellStiffness(0, 1) = tC12;
    mCellStiffness(1, 0) = tC12;
    mCellStiffness(1, 1) = tC11;
    mCellStiffness(2, 2) = tC44;
}

/******************************************************************************//**
 * \brief Linear elastic cubic material model constructor. - 3D
**********************************************************************************/
template<>
Plato::CubicLinearElasticMaterial<3>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
    Plato::LinearElasticMaterial<3>(paramList)
{
    Plato::Scalar tC11 = paramList.get<Plato::Scalar>("C11");
    Plato::Scalar tC12 = paramList.get<Plato::Scalar>("C12");
    Plato::Scalar tC44 = paramList.get<Plato::Scalar>("C44");

    mCellStiffness(0, 0) = tC11;
    mCellStiffness(0, 1) = tC12;
    mCellStiffness(0, 2) = tC12;
    mCellStiffness(1, 0) = tC12;
    mCellStiffness(1, 1) = tC11;
    mCellStiffness(1, 2) = tC12;
    mCellStiffness(2, 0) = tC12;
    mCellStiffness(2, 1) = tC12;
    mCellStiffness(2, 2) = tC11;
    mCellStiffness(3, 3) = tC44;
    mCellStiffness(4, 4) = tC44;
    mCellStiffness(5, 5) = tC44;
}

}
// namespace Plato
