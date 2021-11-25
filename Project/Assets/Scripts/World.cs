using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class World : MonoBehaviour
{
    // Reference to the Prefab. Drag a Prefab into this field in the Inspector.
    public GameObject myPrefab;

    // This script will simply instantiate the Prefab when the game starts.
    void Start()
    {
        // Instantiate at position (0, 0, 0) and zero rotation.
        for(int x = 0; x < 3; x++)
        {
            for (int y = 0; y < 3; y++)
            {
                Debug.Log("instantiate now");
                Instantiate(myPrefab, new Vector3(17 * x, 1, 17 * y), Quaternion.identity).AddComponent<MeshCollider>(); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.

            }

        }
       
    }
}
