using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorldInstantiation : MonoBehaviour
{
    public GameObject myPrefab;

    // Start is called before the first frame update
    void Start(){
        for(int z = 1; z<= 25; z++)
        {
            for (int x = -25; x <= 25; x++)
            {
                for (int y = -25; y <= 25; y++)
                    if(Random.value < 0.5)
                        Instantiate(myPrefab, new Vector3(x, z, y), Quaternion.identity);

            }
        }
    }
}
 