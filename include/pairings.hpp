#pragma once

class PairingUtils
{
public:
    /**
     * @brief Calculates the pairs in a vector which are closest together.
     * Note an element will not pair with itself (not reference wise at least).
     *
     * @tparam T the type to match with
     * @param collection the collection to run this operation on
     * @param comparer a function which should return true iff the second argument is smaller than the third. The function will be given the element which will be the first in the pair, and should return true if the second argument is closer than the third, else it should
     return false.
     * @return std::vector<std::pair<T&, T&>> a vector filled with pairs of nearby elements.
     */
    template <typename T, typename C>
    static std::vector<std::pair<T &, T &>> pair_with_nearest(
        std::vector<T> &collection,
        const std::function<C(const T &, const T &)> &comparer)
    {
        if (collection.size() < 2)
        {
            return std::vector<std::pair<T &, T &>>();
        }

        std::vector<std::pair<T &, T &>> pairs;
        pairs.reserve(collection.size());

        for (size_t i = 0; i < collection.size(); i++)
        {
            size_t smallest = i == 0 ? 1 : 0;
            C smallest_score = comparer(collection.at(i), collection.at(smallest));

            for (size_t j = 0; j < collection.size(); j++)
            {
                if (i != j)
                {
                    C current_score = comparer(collection.at(i), collection.at(j));
                    if (current_score < smallest_score)
                    {
                        smallest = j;
                        smallest_score = current_score;
                    }
                }
            }
            T &first = collection.at(i);
            T &second = collection.at(smallest);
            pairs.push_back({first, second});
        }
        return pairs;
    }
    /**
     * @brief Calculates the pairs in a vector which are closest together.
     * Note an element will not pair with itself (not reference wise at least).
     *
     * @tparam T the type to match with
     * @param collection the collection to run this operation on
     * @param comparer a function which should return true iff the second argument is smaller than the third. The function will be given the element which will be the first in the pair, and should return true if the second argument is closer than the third, else it should
     return false.
     * @return std::vector<std::pair<T&, T&>> a vector filled with pairs of nearby elements.
     */
    template <typename T>
    static std::vector<std::pair<T &, T &>> pair_with_nearest(
        std::vector<T> &collection,
        const std::function<bool(const T &, const T &, const T &)> &comparer)
    {
        if (collection.size() < 2)
        {
            return std::vector<std::pair<T &, T &>>();
        }

        std::vector<std::pair<T &, T &>> pairs;
        pairs.reserve(collection.size());

        for (size_t i = 0; i < collection.size(); i++)
        {
            size_t smallest = i == 0 ? 1 : 0;
            for (size_t j = 0; j < collection.size(); j++)
            {
                if (i != j)
                {
                    if (comparer(collection.at(i), collection.at(j), collection.at(smallest)))
                    {
                        smallest = j;
                    }
                }
            }
            T &first = collection.at(i);
            T &second = collection.at(smallest);
            pairs.push_back({first, second});
        }
        return pairs;
    }

    /**
     * @brief Calculates the pairs in a vector which are closest together.
     * Note an element will not pair with itself (not reference wise at least).
     *
     * @tparam T the type to match with
     * @tparam U the other type to match with
     * @param collection the collection to run this operation on
     * @param comparer a function which should return true iff the second argument is smaller than the third. The function will be given the element which will be the first in the pair, and should return true if the second argument is closer than the third, else it should
     return false.
    * @return std::vector<std::pair<T&, T&>> a vector filled with pairs of nearby elements.
    */
    template <typename T, typename U>
    static std::vector<std::pair<T &, U &>> pair_with_nearest(
        std::vector<T> &collection_a,
        std::vector<U> &collection_b,
        const std::function<bool(const T &, const U &, const U &)> &comparer)
    {
        if (collection_a.size() == 0 || collection_b.size() == 0)
        {
            return std::vector<std::pair<T &, U &>>();
        }

        std::vector<std::pair<T &, U &>> pairs;
        pairs.reserve(collection_a.size());

        for (size_t i = 0; i < collection_a.size(); i++)
        {
            size_t smallest = 0;
            for (size_t j = 0; j < collection_b.size(); j++)
            {
                if (i != j)
                {
                    if (comparer(collection_a.at(i), collection_b.at(j), collection_b.at(smallest)))
                    {
                        smallest = j;
                    }
                }
            }
            T &first = collection_a.at(i);
            U &second = collection_b.at(smallest);
            pairs.push_back({first, second});
        }
        return pairs;
    }

    template <typename T, typename U, typename C>
    static std::vector<std::pair<T &, U &>> pair_with_nearest_value(
        std::vector<T> &collection_a,
        std::vector<U> &collection_b,
        const std::function<C(const T &, const U &)> &comparer)
    {
        if (collection_a.size() < 1 || collection_b.size() < 1)
        {
            return std::vector<std::pair<T &, U &>>();
        }

        std::vector<std::pair<T &, U &>> pairs;
        pairs.reserve(collection_a.size());

        for (size_t i = 0; i < collection_a.size(); i++)
        {
            size_t smallest = 0;
            C smallest_score = comparer(collection_a.at(i), collection_b.at(0));
            for (size_t j = 1; j < collection_b.size(); j++)
            {
                if (i != j)
                {
                    C current_score = comparer(collection_a.at(i), collection_b.at(j));

                    if (current_score < smallest_score)
                    {
                        smallest_score = current_score;
                        smallest = j;
                    }
                }
            }
            T &first = collection_a.at(i);
            U &second = collection_b.at(smallest);
            pairs.push_back({first, second});
        }
        return pairs;
    }
};