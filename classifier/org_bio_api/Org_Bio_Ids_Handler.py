valueMapper={
    "accelaration_aX_g":0,
    "accelaration_aY_g":1,
    "accelaration_aZ_g":2,

    "gyroscope_aX_mdps":3,
    "gyroscope_aY_mdps":4,
    "gyroscope_aZ_mdps":5,

    "magnetometer_aX_mT":6,
    "magnetometer_aY_mT":7,
    "magnetometer_aZ_mT":8,

    "euler_Yaw":9,
    "euler_Roll":10,
    "euler_Pitch":11,

    "quaternion_aW":12,
    "quaternion_aX":13,
    "quaternion_aY":14,
    "quaternion_aZ":15,

    "linearise_acceleration_aX_g":16,
    "linearise_acceleration_aY_g":17,
    "linearise_acceleration_aZ_g":18,

    "gravity_aX_g":19,
    "gravity_aY_g":20,
    "gravity_aZ_g":21
    }

unitMapper = {
    "":0,
    "g":1,
    "mdps":2,
    "deg":3,
    "mT":4
}

valueUnitMapper={
    "accelaration_aX_g":unitMapper["g"],
    "accelaration_aY_g":unitMapper["g"],
    "accelaration_aZ_g":unitMapper["g"],

    "gyroscope_aX_mdps":unitMapper["mdps"],
    "gyroscope_aY_mdps":unitMapper["mdps"],
    "gyroscope_aZ_mdps":unitMapper["mdps"],

    "magnetometer_aX_mT":unitMapper["mT"],
    "magnetometer_aY_mT":unitMapper["mT"],
    "magnetometer_aZ_mT":unitMapper["mT"],

    "euler_Yaw":unitMapper["deg"],
    "euler_Roll":unitMapper["deg"],
    "euler_Pitch":unitMapper["deg"],

    "quaternion_aW":unitMapper[""],
    "quaternion_aX":unitMapper[""],
    "quaternion_aY":unitMapper[""],
    "quaternion_aZ":unitMapper[""],

    "linearise_acceleration_aX_g":unitMapper["g"],
    "linearise_acceleration_aY_g":unitMapper["g"],
    "linearise_acceleration_aZ_g":unitMapper["g"],

    "gravity_aX_g":unitMapper["g"],
    "gravity_aY_g":unitMapper["g"],
    "gravity_aZ_g":unitMapper["g"]
    }



def pk_data_to_org_bio_data(data,mac,uid,deviceTypeId):
    rslt = {
        "id":0,
        "deviceMacAdress":mac,
        "senzorData":[
        ]
    }
    for key in data.keys():
        if(key != "time_ms"):
            rslt["senzorData"].append({
                "uid":uid,
                "timestamps":[int(d*1000) for d in data["time_ms"]],
                "valueID":valueMapper[key],
                "unitID":valueUnitMapper[key],
                "deviceTypeID": deviceTypeId,
                "maxQueueSize": 3000,
                "values":list(data[key])
            })
            # rslt["senzorData"].append({
            #     "uid":uid,
            #     "timestamps":[],
            #     "valueID":valueMapper[key],
            #     "unitID":valueUnitMapper[key],
            #     "deviceTypeID": deviceTypeId,
            #     "maxQueueSize": 3000,
            #     "values":[]
            # })
    return rslt
