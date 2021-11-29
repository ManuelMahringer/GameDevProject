using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerStatus : MonoBehaviour
{
    [SerializeField]
    public int hitpoints = 100;    //0-100HP

    [SerializeField]
    public GameObject[] hotbar;

    [SerializeField]
    public GameObject[,] inventory;

    private int health;

    // Start is called before the first frame update
    void Start()
    {
        health = hitpoints;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void GetHit(){
        
    }

    public void DropInventoryItem(){

    }

    public void AddInventoryItemn(){

    }

    public GameObject CraftNewItem(GameObject[] items){
        return null;
    } 
}
