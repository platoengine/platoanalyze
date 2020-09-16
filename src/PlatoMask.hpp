#pragma once

#include <Omega_h_mesh.hpp>
#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato {

    using OrdinalList = Plato::ScalarVectorT<Plato::OrdinalType>;

    /******************************************************************************/
    /*!
      \brief class for sequence entries
    */
    template <int mSpaceDim>
    class Mask
    /******************************************************************************/
    {

      protected:
        OrdinalList mCellMask;
        OrdinalList mNodeMask;

        /******************************************************************************//**
         * \brief Compute node mask from element mask
         * \param [in] aMesh Omega_h mesh
        **********************************************************************************/
        void
        computeNodeMask(
            Omega_h::Mesh & aMesh
        )
        {
            Kokkos::deep_copy(mNodeMask, 0.0);

            NodeOrdinal<mSpaceDim> tNodeOrdinalFunctor(&aMesh);

            auto tNumCells = mCellMask.extent(0);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),
            LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                if (mCellMask(aCellOrdinal) == 1)
                {
                    for (Plato::OrdinalType tNode=0; tNode<mSpaceDim+1; tNode++)
                    {
                        auto tNodeOrdinal = tNodeOrdinalFunctor(aCellOrdinal, tNode);
                        mNodeMask(tNodeOrdinal) = 1;
                    }
                }
            }, "compute node mask");
        }

        /******************************************************************************//**
         * \brief get location of cell centers in physical space
         * \param [in] aMesh Omega_h mesh
        **********************************************************************************/
        Plato::ScalarMultiVector
        getCellCenters(
            Omega_h::Mesh & aMesh
        )
        {
            Plato::NodeCoordinate<mSpaceDim> tNodeCoordsFunctor(&aMesh);

            auto tNumCells = aMesh.nelems();
            Plato::ScalarMultiVector tCellCenters("cell centers", tNumCells, mSpaceDim);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),
            LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for (Plato::OrdinalType tNode=0; tNode<mSpaceDim+1; tNode++)
                {
                    for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
                    {
                        tCellCenters(aCellOrdinal, tDim) += tNodeCoordsFunctor(aCellOrdinal, tNode, tDim);
                    }
                }
                for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
                {
                    tCellCenters(aCellOrdinal, tDim) /= (mSpaceDim+1);
                }
            }, "get cell centers");

            return tCellCenters;
        }

      public:
        Mask(
            const Omega_h::Mesh & aMesh
        ) :
            mCellMask("cell mask", aMesh.nelems()),
            mNodeMask("node mask", aMesh.nverts()){}

        decltype(mCellMask) cellMask() const {return mCellMask;}
        decltype(mNodeMask) nodeMask() const {return mNodeMask;}

        /******************************************************************************//**
         * \brief Compute node mask from element mask
         * \param [in] aMesh Omega_h mesh
        **********************************************************************************/
        OrdinalList
        getInactiveNodes(
        ) const
        {
            using OrdinalT = Plato::OrdinalType;

            auto tNumEntries = mNodeMask.extent(0);

            // how many zeros in the mask?
            Plato::OrdinalType tSum(0);
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNumEntries),
            LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
            {
              aUpdate += mNodeMask(aOrdinal);
            }, tSum);

            auto tNumFixed = tNumEntries - tSum;
            OrdinalList tNodes("inactive nodes", tNumFixed);

            if (tNumFixed > 0)
            {
                // create a list of nodes with zero mask values
                OrdinalT tOffset(0);
                Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumEntries),
                KOKKOS_LAMBDA (const OrdinalT& iOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
                {
                    const OrdinalT tVal = mNodeMask(iOrdinal);
                    if( tIsFinal ) { tNodes(aUpdate) = iOrdinal; }
                    aUpdate += (1-tVal);
                }, tOffset);
            }

            return tNodes;
        }

    };

    template <int mSpaceDim>
    class BrickMask : public Plato::Mask<mSpaceDim>
    {
      private:
        using Plato::Mask<mSpaceDim>::mCellMask;

        Plato::Scalar mMaximum[mSpaceDim];
        Plato::Scalar mMinimum[mSpaceDim];

        std::string mMaxKeywords[3];
        std::string mMinKeywords[3];

      public:
        /******************************************************************************//**
         * \brief Constructor for Plato::Mask
         * \param [in] aMesh Omega_h mesh
         * \param [in] aInputParams Mask definition
        **********************************************************************************/
        BrickMask(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        ) :
            Plato::Mask<mSpaceDim>(aMesh),
            mMaxKeywords({"Maximum X", "Maximum Y", "Maximum Z"}),
            mMinKeywords({"Minimum X", "Minimum Y", "Minimum Z"})
        {
            for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
            {
                auto tMaxKeyword = mMaxKeywords[tDim];
                if (aInputParams.isType<Plato::Scalar>(tMaxKeyword))
                {
                    mMaximum[tDim] = aInputParams.get<Plato::Scalar>(tMaxKeyword);
                }
                else
                {
                    mMaximum[tDim] = 1e12;
                }
                auto tMinKeyword = mMinKeywords[tDim];
                if (aInputParams.isType<Plato::Scalar>(tMinKeyword))
                {
                    mMinimum[tDim] = aInputParams.get<Plato::Scalar>(tMinKeyword);
                }
                else
                {
                    mMinimum[tDim] = -1e12;
                }
            }

            auto tCellCenters = this->getCellCenters(aMesh);
            
            auto tNumCells = tCellCenters.extent(0);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),
            LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                mCellMask(aCellOrdinal) = 1;

                for (Plato::OrdinalType tDim=0; tDim<mSpaceDim; tDim++)
                {
                    auto tVal = tCellCenters(aCellOrdinal, tDim);
                    if (tVal > mMaximum[tDim]) mCellMask(aCellOrdinal) = 0;
                    if (tVal < mMinimum[tDim]) mCellMask(aCellOrdinal) = 0;
                }
            }, "cell mask");

            this->computeNodeMask(aMesh);
        }
    };

    template <int mSpaceDim>
    class MaskFactory
    {
      public:
        std::shared_ptr<Plato::Mask<mSpaceDim>>
        create(
                  Omega_h::Mesh          & aMesh,
            const Teuchos::ParameterList & aInputParams
        )
        {
            if (!aInputParams.isSublist("Mask"))
            {
                THROWERR("Required parameter list ('Mask') is missing.");
            }
            else
            {
                auto tMaskParams = aInputParams.sublist("Mask");
                if(!tMaskParams.isType<std::string>("Mask Type"))
                {
                    THROWERR("Parsing Mask: Required parameter ('Mask Type') is missing.");
                }
                else
                {
                    auto tMaskType = tMaskParams.get<std::string>("Mask Type");
                    if (tMaskType == "Brick")
                    {
                        return std::make_shared<Plato::BrickMask<mSpaceDim>>(aMesh, tMaskParams);
                    }
                }
            }
        }
    };
} // namespace Plato
