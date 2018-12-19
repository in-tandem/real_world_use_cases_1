def is_prime(number):

    last_digit = number%10

    if last_digit==0 or (number!=2 and last_digit%2==0) or last_digit%5==0:
        return False
    
    factor = len(list(filter(lambda x: number%x==0, range(2,number+1))))

    return True if factor==1 else False

