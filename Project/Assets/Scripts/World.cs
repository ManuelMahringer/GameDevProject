using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class World : MonoBehaviour
{
    // Reference to the Chunk Prefab. Drag a Prefab into this field in the Inspector.
    [SerializeField] 
    private GameObject myPrefab;
    private PhysicMaterial worldMaterial;

    void Awake()
    {
        worldMaterial = new PhysicMaterial
        {
            staticFriction = 0f,
            dynamicFriction = 0f,
            bounciness = 0f,
            frictionCombine = PhysicMaterialCombine.Minimum,
            bounceCombine = PhysicMaterialCombine.Minimum
        };
    }


    void Start()
    {  
        // Instantiate chunks
        for (int x = 0; x < 3; x++)
        {
            for (int y = 0; y < 3; y++)
            {
                Debug.Log("instantiate now");
                MeshCollider mc = Instantiate(myPrefab, new Vector3(17 * x, 1, 17 * y), Quaternion.identity).AddComponent<MeshCollider>(); //  This quaternion corresponds to "no rotation" - the object is perfectly aligned with the world or parent axes.
                mc.material = worldMaterial;
            }

        }

    }
} 